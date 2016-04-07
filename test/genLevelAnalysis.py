import ROOT
ROOT.gROOT.SetBatch(True)
ROOT.gSystem.Load("libFWCoreFWLite.so")
ROOT.gSystem.Load("libTTHCommonClassifier.so")
ROOT.gSystem.Load("libTTHGenLevel.so")
import ROOT.HepMC
import sys, os, pickle

#fastjet setup
ROOT.gSystem.Load("libfastjet")
ROOT.gInterpreter.ProcessLine('#include "fastjet/ClusterSequence.hh"')
ROOT.gInterpreter.ProcessLine("fastjet::ClusterSequence(std::vector<fastjet::PseudoJet>{},fastjet::JetDefinition{});")

from TTH.GenLevel.eventsgen import Events
from TTH.MEAnalysis.MEMUtils import set_integration_vars, add_obj
from PhysicsTools.HeppyCore.framework.analyzer import Analyzer
from PhysicsTools.Heppy.analyzers.core.AutoFillTreeProducer import *

#wrappers for functions we can't call from python
#make a function which increments an iterator, we can't easily do in python
ROOT.gInterpreter.ProcessLine("namespace Util { HepMC::GenEvent::particle_const_iterator next_particle(HepMC::GenEvent::particle_const_iterator it) {it++; return it;}; } ")
ROOT.gInterpreter.ProcessLine("namespace Util { HepMC::GenVertex::particle_iterator next_particle(HepMC::GenVertex::particle_iterator it) {it++; return it;}; } ")
#make a function which dereferences an iterator 
ROOT.gInterpreter.ProcessLine("namespace Util { HepMC::GenParticle* get_particle(HepMC::GenEvent::particle_const_iterator it) {return *it;}; } ")
ROOT.gInterpreter.ProcessLine("namespace Util { HepMC::GenParticle* get_particle(HepMC::GenVertex::particle_iterator it) {return *it;}; } ")

#constants
MIN_PT = 0.0001
JET_MIN_PT_CLUSTERING = 10
JET_MIN_PT = 30
JET_DEF = ROOT.fastjet.JetDefinition(ROOT.fastjet.antikt_algorithm, 0.5)

def lepton_selection(particle):
    """
    Baseline lepton selection
    """
    p4 = particle.p4()
    return p4.Pt() > 30 and abs(p4.Eta())<2.1

#FIXME
def jet_selection(jet):
    """
    Baseline jet selection
    """
    return jet.p4().Pt() > JET_MIN_PT and abs(jet.p4().Eta())<2.1

class Particle(object):

    def __init__(self, physObj):
        self.physObj = physObj

    def p4(self):
        p4 = self.physObj.momentum()
        return ROOT.TLorentzVector(p4.px(), p4.py(), p4.pz(), p4.e())

    def is_final_state(self):
        if not self.physObj.end_vertex() and self.physObj.status() == 1:
            return True
        return False

    def is_me(self):
        if self.physObj.momentum().perp() < MIN_PT:
            return False
        if self.physObj.production_vertex() != None:
            if (self.physObj.production_vertex().id() == 1 and
                self.physObj.status() == 3):
                return True
            elif (self.physObj.production_vertex().id() == 3 and
                self.physObj.status() == 11):
                return True
        return False

    def is_had(self):
        if self.physObj.momentum().perp() < MIN_PT:
            return False
        if self.physObj.production_vertex() != None:
            if (self.physObj.production_vertex().id() == 4 and
                self.physObj.status() == 11):
                return True
        return False

    def is_invisible(self):
        return self.is_final_state() and abs(self.physObj.pdg_id()) in [12, 14, 16]

    def is_lepton(self):
        return abs(self.physObj.pdg_id()) in [11,13]

    def get_parents(self):
        pit = self.physObj.production_vertex().particles_begin(ROOT.HepMC.parents)
        parents = []
        while True:
            pit = ROOT.Util.next_particle(pit)    
            if pit == self.physObj.production_vertex().particles_end(ROOT.HepMC.parents):
                break
            p = Particle(ROOT.Util.get_particle(pit))
            parents += [p]
        return parents

    def __str__(self):
        s = "Particle(pdgId={0}, status={1}, prodVtx={2})".format(
            self.physObj.pdg_id(),
            self.physObj.status(),
            self.physObj.production_vertex().id() if self.physObj.production_vertex() else None
        )
        return s

    def __repr__(self):
        return str(self)

    def eta(self):
        return self.p4().Eta()

    def phi(self):
        return self.p4().Phi()

class GenJet:
    def __init__(self, p4):
        self._p4 = p4

    def p4(self):
        return self._p4

    def eta(self):
        return self.p4().Eta()

    def phi(self):
        return self.p4().Phi()

def cluster_jets(genparticles):
    src_particles = ROOT.std.vector("fastjet::PseudoJet")()
    for x in genparticles:
        p4 = x.physObj.momentum()
        src_particles.push_back(ROOT.fastjet.PseudoJet(
            p4.px(), p4.py(), p4.pz(), p4.e(), 
        ))

    cs = ROOT.fastjet.ClusterSequence(src_particles, JET_DEF)

    ret = []
    for x in sorted(cs.inclusive_jets(JET_MIN_PT_CLUSTERING), key=lambda x: x.pt(), reverse=True):
        gj = GenJet(ROOT.TLorentzVector(x.px(), x.py(), x.pz(), x.e()))

        #fastjet seems to produce spurious jets with pt=0, eta=100000
        if x.pt() > JET_MIN_PT_CLUSTERING:
            ret += [gj]
    return ret

class EventInterpretation(object):

    def __init__(self, **kwargs):
        self.b_quarks = kwargs.get("b_quarks", [])
        self.l_quarks = kwargs.get("l_quarks", [])
        self.leptons = kwargs.get("leptons", [])
        self.invisible = kwargs.get("invisible", [])

    def is_sl(self):
        return len(self.leptons) == 1

    def is_022(self):
        return len(self.b_quarks) >= 4


class ParticleAnalyzer(Analyzer):
    def __init__(self, cfg_ana, cfg_comp, looperName):
        super(ParticleAnalyzer, self).__init__(cfg_ana, cfg_comp, looperName)

    def process(self, event):
        particles = []
        pit = event.input.hepmc_event.particles_begin()
        while True:
            pit = ROOT.Util.next_particle(pit)
            if pit == event.input.hepmc_event.particles_end():
                break
            p = ROOT.Util.get_particle(pit)
            p2 = Particle(p)
            particles += [p2]
        event.particles = particles

        event.interpretations = {}
        return True

class PartonLevelAnalyzer(Analyzer):
    def __init__(self, cfg_ana, cfg_comp, looperName):
        super(PartonLevelAnalyzer, self).__init__(cfg_ana, cfg_comp, looperName)

    def process(self, event):

        event.fstate_particles = filter(lambda x: x.is_final_state(), event.particles)
        event.quarks_me = filter(
            lambda x: x.is_me() and abs(x.physObj.pdg_id()) in [1,2,3,4,5,21], event.particles
        )
        event.b_quarks_me = filter(
            lambda x: abs(x.physObj.pdg_id()) == 5,
            event.quarks_me
        )
        event.l_quarks_me = filter(
            lambda x: abs(x.physObj.pdg_id()) in [1,2,3,4,21],
            event.quarks_me
        )
        event.interpretations["me_parton"] = EventInterpretation(
            b_quarks = [x for x in event.b_quarks_me],
            l_quarks = [x for x in event.l_quarks_me],
        )
        return True

from PhysicsTools.HeppyCore.utils.deltar import matchObjectCollection

class HadronLevelAnalyzer(Analyzer):

    def __init__(self, cfg_ana, cfg_comp, looperName):
        super(HadronLevelAnalyzer, self).__init__(cfg_ana, cfg_comp, looperName)
    
    def process(self, event):    
        
        particles_for_jets = filter(
            lambda x: not x.is_invisible(),
            event.fstate_particles
        )
        leptons = filter(
            lambda x: x.is_lepton(),
            event.fstate_particles
        )
        neutrinos = filter(
            lambda x: x.is_invisible(),
            event.fstate_particles
        )
        leptons_sel = filter(
            lepton_selection,
            leptons
        )

        event.interpretations["hadron"] = EventInterpretation()
        event.interpretations["me_parton"].leptons = [x for x in leptons_sel]
        event.interpretations["hadron"].leptons = [x for x in leptons_sel]

        met = reduce(
            lambda x,y:x+y,
            [x.p4() for x in neutrinos],
            ROOT.TLorentzVector()
        )
        event.interpretations["me_parton"].invisible = met
        event.interpretations["hadron"].invisible = met

        event.particles_for_jets = particles_for_jets
        genjets = cluster_jets(particles_for_jets)
        genjets_sel = filter(
            jet_selection, genjets
        )
        event.jets = genjets_sel
        event.met = met
        event.leptons = leptons_sel

        matches = matchObjectCollection(event.jets, event.quarks_me, 0.3)
        for jet in event.jets:
            jet.matched_me_pdgId = 0
            jet.matched_me_idx = -1

            if matches.has_key(jet) and matches[jet]:
                jet.matched_me_pdgId = matches[jet].physObj.pdg_id()
                jet.matched_me_idx = event.quarks_me.index(matches[jet])
        event.interpretations["hadron"].b_quarks = [x for x in filter(
            lambda x: abs(jet.matched_me_pdgId) == 5, event.jets
        )]
        event.interpretations["hadron"].l_quarks = [x for x in filter(
            lambda x: abs(jet.matched_me_pdgId) in [0,1,2,3,4,21], event.jets
        )]

        event.interpretations_hadron = event.interpretations["hadron"]
        event.interpretations_me = event.interpretations["me_parton"]

        return True

CvectorPSVar = getattr(ROOT, "std::vector<MEM::PSVar::PSVar>")
CvectorPermutations = getattr(ROOT, "std::vector<MEM::Permutations::Permutations>")

import TTH.MEAnalysis.TFClasses as TFClasses
sys.modules["TFClasses"] = TFClasses

def attach_jet_transfer_function(jet, tf_formula):
    """
    Attaches transfer functions to the supplied jet based on the jet eta bin.
    """
    jet_eta_bin = 0
    if abs(jet.p4().Eta())>1.0:
        jet_eta_bin = 1
    jet.tf_b = tf_formula['b'][jet_eta_bin]
    jet.tf_l = tf_formula['l'][jet_eta_bin]
    jet.tf_b.SetNpx(10000)
    jet.tf_b.SetRange(0, 500)

    jet.tf_l.SetNpx(10000)
    jet.tf_l.SetRange(0, 500)

class MEMAnalyzer(Analyzer):

    def __init__(self, cfg_ana, cfg_comp, looperName):
        super(MEMAnalyzer, self).__init__(cfg_ana, cfg_comp, looperName)
        self.cfg = ROOT.MEM.MEMConfig()
        self.cfg.defaultCfg()
 
        self.cplots = ROOT.TFile(os.environ["CMSSW_BASE"]+"/src/TTH/MEAnalysis/root/ControlPlotsV20.root")
        for x,y in [
            ("b", ROOT.MEM.DistributionType.csv_b),
            ("c", ROOT.MEM.DistributionType.csv_c),
            ("l", ROOT.MEM.DistributionType.csv_l),
        ]:
            self.cfg.add_distribution_global(
                y,
                self.cplots.Get(
                    "btagCSV_{0}_pt_eta".format(x)
                )
            )

        pi_file = open(os.environ["CMSSW_BASE"]+"/src/TTH/MEAnalysis/root/transfer_functions.pickle" , 'rb')
        self.tf_matrix = pickle.load(pi_file)
        pi_file.close()
        for nb in [0, 1]:
            for fl1, fl2 in [('b', ROOT.MEM.TFType.bLost), ('l', ROOT.MEM.TFType.qLost)]:
                tf = self.tf_matrix[fl1][nb].Make_CDF()
                tf.SetParameter(0, JET_MIN_PT)
                tf.SetNpx(10000)
                tf.SetRange(0, 500)
                self.cfg.set_tf_global(fl2, nb, tf)
        cfg.transfer_function_method = ROOT.MEM.TFMethod.External


        self.tf_formula = {}
        for fl in ["b", "l"]:
            self.tf_formula[fl] = {}
            for bin in [0, 1]:
                    self.tf_formula[fl][bin] = self.tf_matrix[fl][bin].Make_Formula(False)

        strat = CvectorPermutations()
        strat.push_back(ROOT.MEM.Permutations.QQbarBBbarSymmetry)
        strat.push_back(ROOT.MEM.Permutations.QUntagged)
        strat.push_back(ROOT.MEM.Permutations.BTagged)
        self.cfg.perm_pruning = strat

        self.integrator = ROOT.MEM.Integrand(
            ROOT.MEM.output,
            #ROOT.MEM.output + ROOT.MEM.input + ROOT.MEM.init + ROOT.MEM.init_more + ROOT.MEM.event,
            self.cfg
        )

    def process(self, event):
        for interp_name, interp in event.interpretations.items():
            interp.result_tth = ROOT.MEM.MEMOutput()
            interp.result_ttbb = ROOT.MEM.MEMOutput()

            if interp.is_sl() and interp.is_022():
                print "calling MEM", interp_name, interp.b_quarks
                
                #Create an empty vector for the integration variables
                vars_to_integrate   = CvectorPSVar()
                vars_to_marginalize = CvectorPSVar()

                vars_to_integrate.clear()
                vars_to_marginalize.clear()
                self.integrator.set_cfg(self.cfg)

                set_integration_vars(
                    vars_to_integrate,
                    vars_to_marginalize,
                    ["0w2h2t"]
                )

                for jet in interp.b_quarks:
                    attach_jet_transfer_function(jet, self.tf_formula)
                    add_obj(
                        self.integrator,
                        ROOT.MEM.ObjectType.Jet,
                        p4s=(jet.p4().Pt(), jet.p4().Eta(), jet.p4().Phi(), jet.p4().M()),
                        obs_dict={
                            ROOT.MEM.Observable.BTAG: 1,
                            ROOT.MEM.Observable.CSV: 0,
                            ROOT.MEM.Observable.PDGID: 0,
                            },
                        tf_dict={
                            ROOT.MEM.TFType.bReco: jet.tf_b,
                            ROOT.MEM.TFType.qReco: jet.tf_l,
                        }
                    )
                for lep in interp.leptons:
                    add_obj(
                        self.integrator,
                        ROOT.MEM.ObjectType.Lepton,
                        p4s=(lep.p4().Pt(), lep.p4().Eta(), lep.p4().Phi(), lep.p4().M()),
                        obs_dict={ROOT.MEM.Observable.CHARGE: 1.0 if lep.physObj.pdg_id()>0 else -1.0},
                    )

                print "adding", interp.invisible
                add_obj(
                    self.integrator,
                    ROOT.MEM.ObjectType.MET,
                    p4s=(interp.invisible.Pt(), 0, interp.invisible.Phi(), 0),
                )
                
                results = {}
                for hypo in [
                    ROOT.MEM.Hypothesis.TTH,
                    ROOT.MEM.Hypothesis.TTBB]:
                    fstate = ROOT.MEM.FinalState.LH
                    ret = self.integrator.run(
                        fstate,
                        hypo,
                        vars_to_integrate,
                        vars_to_marginalize
                    )
                    results[hypo] = ret
                interp.result_tth = results[ROOT.MEM.Hypothesis.TTH]
                interp.result_ttbb = results[ROOT.MEM.Hypothesis.TTBB]
                self.integrator.next_event()

        return True

genJetType = NTupleObjectType("genJetType", variables = [
    NTupleVariable("pt", lambda x : x.p4().Pt()),
    NTupleVariable("eta", lambda x : x.p4().Eta()),
    NTupleVariable("phi", lambda x : x.p4().Phi()),
    NTupleVariable("mass", lambda x : x.p4().M()),
    NTupleVariable("matched_me_pdgId", lambda x : x.matched_me_pdgId, type=int),
    NTupleVariable("matched_me_idx", lambda x : x.matched_me_idx, type=int),
])

genLepType = NTupleObjectType("genLepType", variables = [
    NTupleVariable("pt", lambda x : x.p4().Pt()),
    NTupleVariable("eta", lambda x : x.p4().Eta()),
    NTupleVariable("phi", lambda x : x.p4().Phi()),
    NTupleVariable("mass", lambda x : x.p4().M()),
    NTupleVariable("pdgId", lambda x : x.physObj.pdg_id(), type=int),
])

genParticleType = NTupleObjectType("genParticleType", variables = [
    NTupleVariable("pt", lambda x : x.p4().Pt()),
    NTupleVariable("eta", lambda x : x.p4().Eta()),
    NTupleVariable("phi", lambda x : x.p4().Phi()),
    NTupleVariable("mass", lambda x : x.p4().M()),
    NTupleVariable("pdgId", lambda x : x.physObj.pdg_id(), type=int),
    NTupleVariable("status", lambda x : x.physObj.status(), type=int),
])

metType = NTupleObjectType("metType", variables = [
    NTupleVariable("pt", lambda x : x.Pt()),
    NTupleVariable("eta", lambda x : x.Eta()),
    NTupleVariable("phi", lambda x : x.Phi()),
    NTupleVariable("mass", lambda x : x.M()),
])

interp_type = NTupleObjectType("interp_type", variables = [
    NTupleVariable("is_022", lambda x : x.is_022(), type=int),
    NTupleVariable("is_sl", lambda x : x.is_sl(), type=int),
    NTupleVariable("mem_p_tth", lambda x : x.result_tth.p),
    NTupleVariable("mem_p_ttbb", lambda x : x.result_ttbb.p),
])

def fillCoreVariables(self, tr, event, isMC):
    pass

AutoFillTreeProducer.fillCoreVariables = fillCoreVariables

if __name__ == "__main__":

    import PhysicsTools.HeppyCore.framework.config as cfg
    
    particle_ana = cfg.Analyzer(
        ParticleAnalyzer,
        'particle'
    )
    partonlevel_ana = cfg.Analyzer(
        PartonLevelAnalyzer,
        'partonlevel'
    )
    hadlevel_ana = cfg.Analyzer(
        HadronLevelAnalyzer,
        'hadlevel'
    )
    mem_ana = cfg.Analyzer(
        MEMAnalyzer,
        'mem'
    )
    treeProducer = cfg.Analyzer(
        class_object = AutoFillTreeProducer,
        verbose = False,
        vectorTree = True,
        globalVariables = [
            # NTupleVariable(
            #     "numJ",
            #     lambda ev: getattr(ev, "n_bjets_nominal", -1),
            #     help="Number of selected bjets in event"
            # ),
        ],
        globalObjects = {
           "met" : NTupleObject("met", metType, help="Sum of neutrinos"),
            "interpretations_hadron" : NTupleObject("interpretations_hadron", interp_type, help="Hadronic interpretation"),
            "interpretations_me" : NTupleObject("interpretations_me", interp_type, help="ME-level interpretation"),
        },
        collections = {
            "jets" : NTupleCollection("jets", genJetType, 20, help="Generated jets"),
            "particles_for_jets" : NTupleCollection("particles_for_jets", genParticleType, 1000, help="GenParticles for jet clustering"),
            "quarks_me" : NTupleCollection("quarks_me", genParticleType, 20, help="generated quarks at the ME level"),
            "leptons" : NTupleCollection("leps", genLepType, 4, help="Generated leptons"),
        },
    )

    sequence = cfg.Sequence([
        particle_ana,
        partonlevel_ana,
        hadlevel_ana,
        mem_ana,
        treeProducer
    ])

    from PhysicsTools.HeppyCore.framework.services.tfile import TFileService
    output_service = cfg.Service(
        TFileService,
        'outputfile',
        name="outputfile",
        fname='tree.root',
        option='recreate'
    )

    fns = os.environ["FILE_NAMES"].split()
    if len(fns) != 1:
        raise Exception("need only one file")
    dataset = os.environ["DATASETPATH"]
    firstEvent = int(os.environ["SKIP_EVENTS"])
    nEvents = int(os.environ["MAX_EVENTS"])
    
    config = cfg.Config(
        #Run across these inputs
        components = [cfg.Component(
            dataset,
            files = fns,
        )],
        sequence = sequence,
        services = [output_service],
        events_class = Events
    )
    from PhysicsTools.HeppyCore.framework.looper import Looper
    looper = Looper(
        'Loop',
        config,
        nPrint = 0,
        firstEvent = firstEvent,
        nEvents = nEvents
    )
    looper.loop()
    looper.write()
