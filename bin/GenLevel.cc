#include "FWCore/FWLite/interface/AutoLibraryLoader.h"

#include "TROOT.h"
#include "TSystem.h"

#include "HepMC/IO_GenEvent.h"
#include "HepMC/GenEvent.h"
#include "fastjet/ClusterSequence.hh"

int main(int argc, const char** argv) {
    gSystem->Load("libFWCoreFWLite");
    gSystem->Load("libDataFormatsFWLite");

    HepMC::IO_GenEvent ascii_in(
        "/home/joosep/joosep-mac/Downloads/S_stab_2.hepmc2g",
        std::ios::in
    );
    if (ascii_in.rdstate() == std::ios::failbit) {
        std::cerr << "could not read file" << std::endl;
        return 1;
    }

    HepMC::GenEvent* evt = ascii_in.read_next_event();    

    while(evt != nullptr) {
        std::cout << evt << std::endl;
        evt = ascii_in.read_next_event();
    }
    return 0;
}
