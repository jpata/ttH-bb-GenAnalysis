import ROOT
ROOT.gROOT.SetBatch(True)
ROOT.gSystem.Load("libFWCoreFWLite.so")
ROOT.gSystem.Load("libTTHCommonClassifier.so")
ROOT.gSystem.Load("libTTHGenLevel.so")
import ROOT.HepMC

#fastjet setup
ROOT.gSystem.Load("libfastjet")
ROOT.gInterpreter.ProcessLine('#include "fastjet/ClusterSequence.hh"')
ROOT.gInterpreter.ProcessLine("fastjet::ClusterSequence(std::vector<fastjet::PseudoJet>{},fastjet::JetDefinition{});")

from TTH.GenLevel.eventsgen import Events
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
    return jet.p4().Pt() > 30 and abs(jet.p4().Eta())<2.1

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
    for x in sorted(cs.inclusive_jets(), key=lambda x: x.pt(), reverse=True):
        gj = GenJet(ROOT.TLorentzVector(x.px(), x.py(), x.pz(), x.e()))
        ret += [gj]
    return ret

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
        met = reduce(
            lambda x,y:x+y,
            [x.p4() for x in neutrinos],
            ROOT.TLorentzVector()
        )
    
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
            if matches.has_key(jet) and matches[jet]:
                jet.matched_me_pdgId = matches[jet].physObj.pdg_id()

        return True

genJetType = NTupleObjectType("genJetType", variables = [
    NTupleVariable("pt", lambda x : x.p4().Pt()),
    NTupleVariable("eta", lambda x : x.p4().Eta()),
    NTupleVariable("phi", lambda x : x.p4().Phi()),
    NTupleVariable("mass", lambda x : x.p4().M()),
    NTupleVariable("matched_me_pdgId", lambda x : x.matched_me_pdgId, type=int),
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

        },
        collections = {
            "jets" : NTupleCollection("jets", genJetType, 20, help="Generated jets", mcOnly=False),
            "particles_for_jets" : NTupleCollection("particles_for_jets", genParticleType, 1000, help="GenParticles for jet clustering", mcOnly=False),
            "quarks_me" : NTupleCollection("quarks_me", genParticleType, 20, help="generated quarks at the ME level", mcOnly=False),
            "leptons" : NTupleCollection("leps", genLepType, 4, help="Generated leptons", mcOnly=False),
        }
    )

    sequence = cfg.Sequence([
        particle_ana,
        partonlevel_ana,
        hadlevel_ana,
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

    config = cfg.Config(
        #Run across these inputs
        components = [cfg.Component(
            "S_dec_had",
            files = ["/home/joosep/joosep-mac/Downloads/S_dec_had_1.hepmc2g"],
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
        nEvents = 500
    )
    looper.loop()
    looper.write()
