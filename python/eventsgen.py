import sys, ROOT

class Events(object):

    def __init__(self, filenames, treename):
        if len(filenames) != 1:
            raise ValueError("HepMC Events only works with one file name")

        self.hepmc_in = ROOT.HepMC.IO_GenEvent(
            filenames[0],
            getattr(ROOT.std.ios, "in")
        )
        self.hepmc_event = self.hepmc_in.read_next_event()
        self.iEv_cur = 0

    def __len__(self):
        return sys.maxint

    def __getitem__(self, index):
        return self

    def __iter__(self):
        return self

    def next(self):
        if self.hepmc_event == None:
            raise StopIteration
        else:
            self.hepmc_event = self.hepmc_in.read_next_event()
            self.iEv_cur += 1
            return self

    def __getitem__(self, iEv):
        while self.iEv_cur < iEv:
            self.next()
        return self
