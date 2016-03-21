import ROOT
ROOT.gSystem.Load("libFWCoreFWLite.so")
ROOT.gSystem.Load("libTTHCommonClassifier.so")
ROOT.gSystem.Load("libTTHGenLevel.so")
import ROOT.HepMC

Particles = ROOT.Particles()

if __name__ == "__main__":

    evt = ROOT.HepMC.IO_GenEvent(
        "/home/joosep/joosep-mac/Downloads/S_stab_2.hepmc2g",
        getattr(ROOT.std.ios, "in")
    )
    while True:
        ev = evt.read_next_event()

        particles = Particles.get_particles(ev)
        for p in particles:
            print p
        if ev == None:
            break
