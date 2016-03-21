#include "TTH/GenLevel/interface/Utility.h"

bool TTHGenLevel::Utility::is_final_state( const HepMC::GenParticle* p ) {
    if ( !p->end_vertex() && p->status()==1 ) return 1;
    return 0;
}

std::vector<HepMC::GenParticle> TTHGenLevel::Utility::GenEvent_get_particles(HepMC::GenEvent* evt) {
    std::vector<HepMC::GenParticle> ret;

    for (auto p = evt->particles_begin();
        p != evt->particles_end(); ++p ) {
        ret.push_back(**p);
    }
    return ret;
}
