#include "TTH/GenLevel/interface/Utility.h"

bool TTHGenLevel::Particles::is_final_state( const HepMC::GenParticle* p ) {
    if ( !p->end_vertex() && p->status()==1 ) return 1;
    return 0;
}

std::vector<HepMC::GenParticle> TTHGenLevel::Particles::get_vector(HepMC::GenEvent* evt) {
    std::vector<HepMC::GenParticle> ret;

    for (auto p = evt->particles_begin();
        p != evt->particles_end(); ++p ) {
        ret.push_back(**p);
    }
    return ret;
}
