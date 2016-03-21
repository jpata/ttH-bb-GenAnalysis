import ROOT
ROOT.gSystem.Load("libFWCoreFWLite.so")
ROOT.gSystem.Load("libTTHCommonClassifier.so")
ROOT.gSystem.Load("libTTHGenLevel.so")
import ROOT.HepMC

ROOT.gSystem.Load("libfastjet")
ROOT.gInterpreter.ProcessLine('#include "fastjet/ClusterSequence.hh"')

jet_def = ROOT.fastjet.JetDefinition(ROOT.fastjet.antikt_algorithm, 0.7)
ROOT.gInterpreter.ProcessLine("fastjet::ClusterSequence(std::vector<fastjet::PseudoJet>{},fastjet::JetDefinition{});")

from PhysicsTools.HeppyCore.framework.eventsgen import Events

class Particle(ROOT.HepMC.GenParticle):
    def is_final_state(self):
        if self.end_vertex() == None and self.status() == 1:
            return True
        return False

class GenJet:
    def __init__(self, p4):
        self.p4 = p4

def cluster_jets(genparticles):
    src_particles = ROOT.std.vector("fastjet::PseudoJet")()
    for x in genparticles:
        p4 = x.momentum()
        src_particles.push_back(ROOT.fastjet.PseudoJet(
            p4.px(), p4.py(), p4.pz(), p4.e(), 
        ))

    cs = ROOT.fastjet.ClusterSequence(src_particles, jet_def)

    ret = []
    for x in sorted(cs.inclusive_jets(), key=lambda x: x.pt(), reverse=True):
        gj = GenJet(ROOT.TLorentzVector(x.px(), x.py(), x.pz(), x.e()))
        ret += [gj]
    return ret

if __name__ == "__main__":

    evts = Events("/home/joosep/joosep-mac/Downloads/S_dec_had_1.hepmc2g")
    for ev in evts:
        particles = ROOT.TTHGenLevel.Utility.GenEvent_get_particles(ev.hepmc_event)
        particles = map(Particle, particles)
        fstate_particles = filter(lambda x: x.is_final_state(), particles)

        genjets = cluster_jets(fstate_particles)
        print genjets
        if ev == None:
            break
