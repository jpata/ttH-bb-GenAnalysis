import ROOT
ROOT.gROOT.SetBatch(True)
ROOT.gSystem.Load("libFWCoreFWLite.so")
ROOT.gSystem.Load("libTTHCommonClassifier.so")
ROOT.gSystem.Load("libTTHGenLevel.so")
import ROOT.HepMC
import sys, os, pickle
import logging
import numpy as np

#fastjet setup
ROOT.gSystem.Load("libfastjet")
ROOT.gInterpreter.ProcessLine('#include "fastjet/ClusterSequence.hh"')
ROOT.gInterpreter.ProcessLine("fastjet::ClusterSequence(std::vector<fastjet::PseudoJet>{},fastjet::JetDefinition{});")

from TTH.GenLevel.eventsgen import Events
from TTH.MEAnalysis.MEMUtils import set_integration_vars, add_obj
from TTH.MEAnalysis.MEMConfig import MEMConfig

from PhysicsTools.HeppyCore.framework.analyzer import Analyzer
from PhysicsTools.Heppy.analyzers.core.AutoFillTreeProducer import *
from PhysicsTools.HeppyCore.utils.deltar import matchObjectCollection

CvectorPSVar = getattr(ROOT, "std::vector<MEM::PSVar::PSVar>")
CvectorPermutations = getattr(ROOT, "std::vector<MEM::Permutations::Permutations>")

#wrappers for functions we can't call from python
#make a function which increments an iterator, we can't easily do in python
ROOT.gInterpreter.ProcessLine("namespace Util { HepMC::GenEvent::particle_const_iterator next_particle(HepMC::GenEvent::particle_const_iterator it) {it++; return it;}; } ")
ROOT.gInterpreter.ProcessLine("namespace Util { HepMC::GenVertex::particle_iterator next_particle(HepMC::GenVertex::particle_iterator it) {it++; return it;}; } ")
#make a function which dereferences an iterator 
ROOT.gInterpreter.ProcessLine("namespace Util { HepMC::GenParticle* get_particle(HepMC::GenEvent::particle_const_iterator it) {return *it;}; } ")
ROOT.gInterpreter.ProcessLine("namespace Util { HepMC::GenParticle* get_particle(HepMC::GenVertex::particle_iterator it) {return *it;}; } ")

#Needed to correctly unpickle, otherwise
#   File "/opt/cms/slc6_amd64_gcc530/lcg/root/6.06.00-ikhhed4/lib/ROOT.py", line 303, in _importhook
#     return _orig_ihook( name, *args, **kwds )
# ImportError: No module named TFClasses
import TTH.MEAnalysis.TFClasses as TFClasses
sys.modules["TFClasses"] = TFClasses

pi_file = open(os.environ["CMSSW_BASE"]+"/src/TTH/MEAnalysis/data/transfer_functions.pickle" , 'rb')
tf_matrix = pickle.load(pi_file)
pi_file.close()

class Conf(dict):
    def __getattr__(self, x):
        return self[x]

conf = Conf()
conf["jets"] = {
    "pt": 20,
    "eta": 2.4,
    "pt_clustering": 10,
    "def":  "ROOT.fastjet.JetDefinition(ROOT.fastjet.antikt_algorithm, 0.5)"
}

conf["leptons"] = {
    "pt": 30,
    "eta": 2.4
}

conf["mem"] = {
"n_integration_points_mult": 1.0
}

conf["tf_matrix"] = tf_matrix

def lepton_selection(particle, conf=conf):
    """
    Baseline lepton selection
    """
    p4 = particle.p4()
    return p4.Pt() > conf["leptons"]["pt"] and abs(p4.Eta())<conf["leptons"]["eta"]

#FIXME
def jet_selection(jet, conf=conf):
    """
    Baseline jet selection
    """
    return jet.p4().Pt() > conf["jets"]["pt"] and abs(jet.p4().Eta())<conf["jets"]["eta"]

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

def cluster_jets(genparticles, conf=conf):
    src_particles = ROOT.std.vector("fastjet::PseudoJet")()
    for x in genparticles:
        p4 = x.physObj.momentum()
        src_particles.push_back(ROOT.fastjet.PseudoJet(
            p4.px(), p4.py(), p4.pz(), p4.e(), 
        ))

    cs = ROOT.fastjet.ClusterSequence(src_particles, eval(conf["jets"]["def"]))

    ret = []
    for x in sorted(cs.inclusive_jets(conf["jets"]["pt_clustering"]), key=lambda x: x.pt(), reverse=True):
        gj = GenJet(ROOT.TLorentzVector(x.px(), x.py(), x.pz(), x.e()))

        #fastjet seems to produce spurious jets with pt=0, eta=100000
        if x.pt() > conf["jets"]["pt_clustering"]:
            ret += [gj]
    return ret

class EventInterpretation(object):
    """Summarizes an event at the hadron level, as required for MEM evaluation
    
    Attributes:
        b_quarks (list): All the considered b-quarks
        hypo (TYPE): hypothesis
        invisible (TLorenzVector): the MET 
        l_quarks (list): All the considered light quarks
        leptons (list): All the considered leptons
    """

    SELECTION_FUNCTIONS = {
        "sl_2w2h2t": lambda i: i.is_sl() and i.is_2w2h2t(),
        "sl_1w2h2t": lambda i: i.is_sl() and i.is_1w2h2t(),
        "sl_0w2h2t": lambda i: i.is_sl() and i.is_0w2h2t(),
        "dl_0w2h2t": lambda i: i.is_dl() and i.is_0w2h2t(),
    }
    INTEGRATION_VARS = {
        "sl_0w2h2t": [
            ROOT.MEM.PSVar.cos_q1, ROOT.MEM.PSVar.phi_q1, ROOT.MEM.PSVar.cos_qbar1, ROOT.MEM.PSVar.phi_qbar1
        ],
        "sl_1w2h2t": [
            ROOT.MEM.PSVar.cos_q1, ROOT.MEM.PSVar.phi_q1
        ]
    }

    def __init__(self, **kwargs):
        self.b_quarks = kwargs.get("b_quarks", [])
        self.l_quarks = kwargs.get("l_quarks", [])
        self.leptons = kwargs.get("leptons", [])
        self.invisible = kwargs.get("invisible", ROOT.TLorentzVector())
        self.hypo = kwargs.get("hypo")
        self.selection_function = self.SELECTION_FUNCTIONS[kwargs.get("selection_function")]


        if len(self.leptons) == 0:
            self.fstate = ROOT.MEM.FinalState.HH
        elif len(self.leptons) == 1:
            self.fstate = ROOT.MEM.FinalState.LH
        elif len(self.leptons) == 2:
            self.fstate = ROOT.MEM.FinalState.LL

        self.vars_to_integrate   = CvectorPSVar()
        self.vars_to_marginalize = CvectorPSVar()

        for integ in self.INTEGRATION_VARS.get(kwargs.get("selection_function"), []):
            self.vars_to_integrate.push_back(integ)

        self.result_tth = ROOT.MEM.MEMOutput()
        self.result_ttbb = ROOT.MEM.MEMOutput()
        self.was_calculated = False


    def set_variables(self):
        if self.is_sl() and self.hypo in ["0w2h2t", "1w2h2t"]:
            set_integration_vars(
                self.vars_to_integrate,
                self.vars_to_marginalize,
                [self.hypo]
            )

    def is_sl(self):
        return len(self.leptons) == 1

    def is_dl(self):
        return len(self.leptons) == 2

    def is_0w2h2t(self):
        return len(self.b_quarks) >= 4

    def is_2w2h2t(self):
        return len(self.b_quarks) >= 4 and len(self.l_quarks) >= 2

    def is_1w2h2t(self):
        return len(self.b_quarks) >= 4 and len(self.l_quarks) >= 1

    def __str__(self):
        s = "Interpretation\n"
        s += "  Nb={0} Nq={1} Nl={2}\n".format(
            len(self.b_quarks),
            len(self.l_quarks),
            len(self.leptons)
        )
        s += "  hypo={0}\n".format(self.hypo)
        for q in self.b_quarks:
            s += "  b: {0}\n".format(q)
        for q in self.l_quarks:
            s += "  q: {0}\n".format(q)
        for q in self.leptons:
            s += "  l: {0}\n".format(q)
        return s
            
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
                jet.matched_me_pdgId = matches[jet].pdg_id()
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
        self.conf = cfg_ana._conf 
        self.cplots = ROOT.TFile(os.environ["CMSSW_BASE"]+"/src/TTH/MEAnalysis/data/ControlPlotsV20.root")

        self.tf_formula = {}
        for fl in ["b", "l"]:
            self.tf_formula[fl] = {}
            for bin in [0, 1]:
                    self.tf_formula[fl][bin] = self.conf["tf_matrix"][fl][bin].Make_Formula(False)

        self.default_cfg = self.setup_cfg_default(self.conf)
        self.integrator = ROOT.MEM.Integrand(
            0,
            self.default_cfg.cfg
        )
        self.logger = logging.getLogger("MEMAnalyzer")

    @staticmethod
    def setup_cfg_default(conf):
        cfg = MEMConfig(conf)

        cfg.configure_transfer_function(conf)

        strat = CvectorPermutations()
        strat.push_back(ROOT.MEM.Permutations.QQbarBBbarSymmetry)
        strat.push_back(ROOT.MEM.Permutations.QUntagged)
        strat.push_back(ROOT.MEM.Permutations.BTagged)
        cfg.cfg.perm_pruning = strat
        return cfg

    def process(self, event):

        #Process all event interpretations
        for interp_name, interp in event.interpretations.items():
            self.logger.info("process: inter={0} sel={1} {2},{3},{4} {5}".format(
                interp_name,
                interp.selection_function(interp),
                len(interp.b_quarks), len(interp.l_quarks), len(interp.leptons),
                interp.hypo
            ))

            if not interp.selection_function(interp):
                continue
            self.logger.info("process: interp={0}".format(str(interp)))

            interp.result_tth = ROOT.MEM.MEMOutput()
            interp.result_ttbb = ROOT.MEM.MEMOutput()

            self.integrator.set_cfg(getattr(interp, "mem_cfg", self.default_cfg).cfg)

            for jet in interp.b_quarks:
                if hasattr(jet, "tf"):
                    jet.tf_b = jet.tf
                    jet.tf_l = jet.tf
                else:
                    attach_jet_transfer_function(jet, self.tf_formula)
                add_obj(
                    self.integrator,
                    ROOT.MEM.ObjectType.Jet,
                    p4s=(jet.p4().Pt(), jet.p4().Eta(), jet.p4().Phi(), jet.p4().M()),
                    obs_dict={
                        ROOT.MEM.Observable.BTAG: 1,
                        },
                    tf_dict={
                        ROOT.MEM.TFType.bReco: jet.tf_b,
                        ROOT.MEM.TFType.qReco: jet.tf_l,
                    }
                )
            for jet in interp.l_quarks:
                if hasattr(jet, "tf"):
                    jet.tf_b = jet.tf
                    jet.tf_l = jet.tf
                else:
                    attach_jet_transfer_function(jet, self.tf_formula)
                add_obj(
                    self.integrator,
                    ROOT.MEM.ObjectType.Jet,
                    p4s=(jet.p4().Pt(), jet.p4().Eta(), jet.p4().Phi(), jet.p4().M()),
                    obs_dict={
                        ROOT.MEM.Observable.BTAG: 0,
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
                    obs_dict={ROOT.MEM.Observable.CHARGE: 1.0 if lep.pdg_id()>0 else -1.0},
                )

            add_obj(
                self.integrator,
                ROOT.MEM.ObjectType.MET,
                p4s=(interp.invisible.Pt(), 0, interp.invisible.Phi(), 0),
            )
            
            results = {}
            for me_hypo in [
                ROOT.MEM.Hypothesis.TTH,
                ROOT.MEM.Hypothesis.TTBB]:
                self.logger.debug("Evaluating hypo={0}".format(me_hypo))
                #ret = ROOT.MEM.MEMOutput() 
                ret = self.integrator.run(
                    interp.fstate,
                    me_hypo,
                    interp.vars_to_integrate,
                    interp.vars_to_marginalize
                )
                self.logger.debug("p={0}".format(ret.p))
                results[me_hypo] = ret
            interp.was_calculated = True
            interp.result_tth = results[ROOT.MEM.Hypothesis.TTH]
            interp.result_ttbb = results[ROOT.MEM.Hypothesis.TTBB]
            self.integrator.next_event()

        return True

genJetType = NTupleObjectType("genJetType", variables = [
    NTupleVariable("pt", lambda x : x.p4().Pt()),
    NTupleVariable("eta", lambda x : x.p4().Eta()),
    NTupleVariable("phi", lambda x : x.p4().Phi()),
    NTupleVariable("mass", lambda x : x.p4().M()),
    NTupleVariable("pdgId", lambda x : x.pdg_id(), the_type=int),
    NTupleVariable("status", lambda x : x.status(), the_type=int),
    NTupleVariable("numBHadrons", lambda x: getattr(x, "numBHadrons", -99), the_type=int),
    NTupleVariable("numCHadrons", lambda x: getattr(x, "numCHadrons", -99), the_type=int),
])

genLepType = NTupleObjectType("genLepType", variables = [
    NTupleVariable("pt", lambda x : x.p4().Pt()),
    NTupleVariable("eta", lambda x : x.p4().Eta()),
    NTupleVariable("phi", lambda x : x.p4().Phi()),
    NTupleVariable("mass", lambda x : x.p4().M()),
    NTupleVariable("pdgId", lambda x : x.pdg_id(), the_type=int),
])

genParticleType = NTupleObjectType("genParticleType", variables = [
    NTupleVariable("pt", lambda x : x.p4().Pt()),
    NTupleVariable("eta", lambda x : x.p4().Eta()),
    NTupleVariable("phi", lambda x : x.p4().Phi()),
    NTupleVariable("mass", lambda x : x.p4().M()),
    NTupleVariable("pdgId", lambda x : x.pdg_id(), the_type=int),
    NTupleVariable("status", lambda x : x.status(), the_type=int),
])

metType = NTupleObjectType("metType", variables = [
    NTupleVariable("pt", lambda x : x.Pt()),
    NTupleVariable("eta", lambda x : x.Eta()),
    NTupleVariable("phi", lambda x : x.Phi()),
    NTupleVariable("mass", lambda x : x.M()),
])

interp_type = NTupleObjectType("interp_type", variables = [
    NTupleVariable("was_calculated", lambda x : x.was_calculated, the_type=int),
    NTupleVariable("n_b", lambda x : len(x.b_quarks), the_type=int),
    NTupleVariable("n_q", lambda x : len(x.l_quarks), the_type=int),
    NTupleVariable("n_l", lambda x : len(x.leptons), the_type=int),

    NTupleVariable("mem_p_tth", lambda x : x.result_tth.p),
    NTupleVariable("mem_p_ttbb", lambda x : x.result_ttbb.p),
    NTupleVariable("mem_num_perm_tth", lambda x : x.result_tth.num_perm),
    NTupleVariable("mem_num_perm_ttbb", lambda x : x.result_ttbb.num_perm),
    NTupleVariable("mem_time_tth", lambda x : x.result_tth.time),
    NTupleVariable("mem_time_ttbb", lambda x : x.result_ttbb.time),
])

def fillCoreVariables(self, tr, event, isMC):
    if isMC:
        for x in ["run", "lumi", "evt", "xsec", "genWeight"]:
            tr.fill(x, getattr(event.input, x))
    else:
        for x in ["run", "lumi", "evt"]:
            tr.fill(x, getattr(event.input, x))
    #tr.fill("isData", not isMC)
    #tr.fill("intLumi", 0)

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
