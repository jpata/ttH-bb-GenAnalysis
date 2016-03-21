
#ifndef MEMCLASSIFIER_H
#define MEMCLASSIFIER_H

#include "HepMC/IO_GenEvent.h"
#include "HepMC/GenEvent.h"
#include "fastjet/ClusterSequence.hh"

namespace TTHGenLevel {
class Particles {
public:
    static bool is_final_state( const HepMC::GenParticle* p );
    static std::vector<HepMC::GenParticle> get_vector(HepMC::GenEvent* evt);

    Particles() {
    
    }
};
}

#endif
