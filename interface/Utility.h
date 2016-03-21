
#ifndef MEMCLASSIFIER_H
#define MEMCLASSIFIER_H

#include "HepMC/IO_GenEvent.h"
#include "HepMC/GenEvent.h"
#include "fastjet/ClusterSequence.hh"

namespace TTHGenLevel {
class Utility {
public:
    static bool is_final_state( const HepMC::GenParticle* p );
    static std::vector<HepMC::GenParticle> GenEvent_get_particles(HepMC::GenEvent* evt);

    Utility() {
    
    }
    ~Utility() {
    
    }
};
}

#endif
