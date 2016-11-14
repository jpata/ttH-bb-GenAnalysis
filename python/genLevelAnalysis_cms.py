import ROOT
ROOT.gROOT.SetBatch(True)
ROOT.gSystem.Load("libFWCoreFWLite.so")
import sys, os, pickle
import logging
import copy
import numpy as np

from PhysicsTools.HeppyCore.framework.chain import Chain as Events
from TTH.MEAnalysis.MEMUtils import set_integration_vars, add_obj
from TTH.MEAnalysis.samples_base import getSitePrefix
from PhysicsTools.HeppyCore.framework.analyzer import Analyzer
from PhysicsTools.Heppy.analyzers.core.AutoFillTreeProducer import *
from TTH.GenLevel.genLevelAnalysis import MEMAnalyzer, EventInterpretation, interp_type, conf, genParticleType
from PhysicsTools.HeppyCore.utils.deltar import matchObjectCollection
from PhysicsTools.HeppyCore.framework.analyzer import Analyzer
from TTH.MEAnalysis.VHbbTree import *

import sys, types
def get_refcounts():
    d = {}
    sys.modules
    # collect all classes
    for m in sys.modules.values():
        for sym in dir(m):
            o = getattr (m, sym)
            if type(o) is types.ClassType:
                d[o] = sys.getrefcount (o)
    # sort by refcount
    pairs = map (lambda x: (x[1],x[0]), d.items())
    pairs.sort()
    pairs.reverse()
    return pairs

def print_top_100():
    print "top 100 refs"
    for n, c in get_refcounts()[:100]:
        print '%10d %s' % (n, c.__name__)

class EventAnalyzer(Analyzer):
    def __init__(self, cfg_ana, cfg_comp, looperName):
        super(EventAnalyzer, self).__init__(cfg_ana, cfg_comp, looperName)
    def process(self, event):
        event.GenBQuarkFromH = GenBQuarkFromH.make_array(event.input)
        event.GenBQuarkFromHafterISR = GenBQuarkFromHafterISR.make_array(event.input)
        event.GenBQuarkFromTop = GenBQuarkFromTop.make_array(event.input)
        event.GenHiggsBoson = GenHiggsBoson.make_array(event.input)
        event.GenJet = GenJet.make_array(event.input)
        event.GenLep = GenLep.make_array(event.input)
        event.GenLepFromTop = GenLepFromTop.make_array(event.input)
        event.GenNuFromTop = GenNuFromTop.make_array(event.input)
        event.GenStatus2bHad = GenStatus2bHad.make_array(event.input)
        event.GenTop = GenTop.make_array(event.input)
        event.GenWZQuark = GenWZQuark.make_array(event.input)
        event.ttCls = getattr(event.input, "ttCls", None)

class Particle:
    @staticmethod
    def from_obj(obj, parent_id=0):
        p4 = ROOT.TLorentzVector()
        p4.SetPtEtaPhiM(obj.pt, obj.eta, obj.phi, obj.mass)
        return Particle(
            _p4=p4,
            _pdg_id = obj.pdgId,
            _status = obj.status,
            _parent_id = parent_id
        )

    def __init__(self, **kwargs):
        self._p4 = kwargs.get("_p4") 
        self._pdg_id = kwargs.get("_pdg_id") 
        self._status = kwargs.get("_status") 
        self._parent_id = kwargs.get("_parent_id") 
        #self.__dict__.update(kwargs)
        #self.physObj = self

    def is_final_state(self):
        return True

    def pdg_id(self):
        return self._pdg_id

    def p4(self):
        return self._p4

    def eta(self):
        return self.p4().Eta()

    def phi(self):
        return self.p4().Phi()

    def status(self):
        return self._status

    def parent_id(self):
        return self._parent_id

    def __str__(self):
        s = "Particle(pt={0:.2f}, eta={1:.2f}, phi={2:.2f}, id={3}, pid={4})".format(
            self.p4().Pt(), self.p4().Eta(), self.p4().Phi(),
            self.pdg_id(), self.parent_id()
        )
        return s


class EventRepresentation:
    def __init__(self, **kwargs):
        self.b_quarks = kwargs.get("b_quarks", [])
        self.l_quarks = kwargs.get("l_quarks", [])
        self.leptons = kwargs.get("leptons", [])
        self.invisible = kwargs.get("invisible", ROOT.TLorentzVector())

    def perturb_momentum(self, scale={}, res={}):
        ks = ["b_quarks", "l_quarks", "leptons"]
        for k in ks:
            sf_scale = scale.get(k, 0.0)
            sig_res = res.get(k, 0.0)

            for iobj, obj in enumerate(getattr(self, k)):
                v_res = 0.0

                if sig_res > 0:
                    v_res = np.random.normal(0.0, sig_res)

                pt_new = (1.0 + sf_scale + v_res) * obj.p4().Pt()
                m_new = (1.0 + sf_scale + v_res) * obj.p4().M()
                obj.p4().SetPtEtaPhiM(pt_new, obj.p4().Eta(), obj.p4().Phi(), m_new)

    def perturb_flavour(self, effs):

        b_quarks = []
        l_quarks = []

        tag_cands = []
        for q in self.b_quarks:
            tag_cands += [(q, "b")]

        for q in self.l_quarks:
            tag_cands += [(q, "l")]

        for q, fl in tag_cands:
            is_tagged = np.random.uniform() < effs[fl]
            if is_tagged:
                b_quarks += [q]
            else:
                l_quarks += [q]

        self.b_quarks = b_quarks
        self.l_quarks = l_quarks

    def make_interpretation(self, hypo, conf):
        if hypo == "0w2h2t":
            interp = EventInterpretation(
                b_quarks = self.b_quarks,
                leptons = self.leptons,
                invisible = self.invisible,
                hypo = "0w2h2t"
            )
        elif hypo == "2w2h2t":
            interp = EventInterpretation(
                b_quarks = self.b_quarks,
                l_quarks = self.l_quarks,
                leptons = self.leptons,
                invisible = self.invisible,
                hypo = "2w2h2t"
            )

        interp.mem_cfg = "default"
        return interp

class GenQuarkLevelAnalyzer(Analyzer):
    def __init__(self, cfg_ana, cfg_comp, looperName):
        super(GenQuarkLevelAnalyzer, self).__init__(cfg_ana, cfg_comp, looperName)
        self.conf = cfg_ana._conf
        self.logger = logging.getLogger("GenQuarkLevelAnalyzer")

    def process(self, event):
        event.gen_b_h = map(lambda p: Particle.from_obj(p, 25), event.GenBQuarkFromH)
        event.gen_b_t = map(lambda p: Particle.from_obj(p, 6), event.GenBQuarkFromTop)
        event.gen_b_others = map(lambda p: Particle.from_obj(p, 0), event.GenStatus2bHad)
        event.gen_q_w = map(lambda p: Particle.from_obj(p, 24), event.GenWZQuark)
        event.gen_lep = map(lambda p: Particle.from_obj(p, 6), event.GenLepFromTop)
        event.gen_nu = map(lambda p: Particle.from_obj(p, 6), event.GenNuFromTop)
        event.gen_met = sum([p.p4() for p in event.gen_nu], ROOT.TLorentzVector())
        self.logger.info("process "
            "nGenBother={0} nGenBH={1} nGenBt={2} nGenLep={3}".format(
                len(event.gen_b_others), len(event.gen_b_h), len(event.gen_b_t), len(event.gen_lep)
            )
        )
        event.interpretations = {}

        ###
        ### b-quarks
        ###
        b_hadrons = event.gen_b_h + event.gen_b_t
        self.logger.info("b from h: {0}".format(map(str, event.gen_b_h)))
        self.logger.info("b from t: {0}".format(map(str, event.gen_b_t)))

        #Find b-hadrons in additional set that are not matched to existing higgs or top
        #derived b-hadrons
        additional_b_hadrons = []
        for bhad in event.gen_b_others:
            matches = matchObjectCollection(event.gen_b_others, b_hadrons, 0.3)
            if not matches.has_key(bhad) or matches[bhad] is None:
                self.logger.info("no match for pt={0}, adding".format(bhad.p4().Pt()))
                b_hadrons += [bhad]
                additional_b_hadrons += [bhad]
        self.logger.info("b from other: {0}".format(map(str, additional_b_hadrons)))

        event.gen_b_others_cleaned = additional_b_hadrons

        #Apply kinematic cuts and take up to first 4
        event.b_hadrons_sorted = sorted(b_hadrons, key=lambda x: x.p4().Pt(), reverse=True)
        event.b_hadrons_sorted = filter(lambda x: x.p4().Pt() > self.conf["jets"]["pt"], event.b_hadrons_sorted)
        event.b_hadrons_sorted = filter(lambda x: abs(x.p4().Eta()) < self.conf["jets"]["eta"], event.b_hadrons_sorted)
        event.b_hadrons_sorted = event.b_hadrons_sorted

        event.repr_quarks = EventRepresentation(
            b_quarks = event.b_hadrons_sorted[:6],
            l_quarks = event.gen_q_w,
            leptons = event.gen_lep,
            invisible = event.gen_met,
        )

        event.interpretations["gen_b_quark"] = event.repr_quarks.make_interpretation("0w2h2t", self.conf)
        event.interpretations["gen_b_q_quark"] = event.repr_quarks.make_interpretation("2w2h2t", self.conf)

class GenJetLevelAnalyzer(Analyzer):
    def __init__(self, cfg_ana, cfg_comp, looperName):
        super(GenJetLevelAnalyzer, self).__init__(cfg_ana, cfg_comp, looperName)
        self.conf = cfg_ana._conf
        self.logger = logging.getLogger("GenJetLevelAnalyzer")

    def process(self, event):
        #get all the gen-jet
        event.gen_jet = map(lambda p: Particle.from_obj(p, 0), event.GenJet)

        self.logger.info("process: gen_jets={0}".format(map(str, event.gen_jet)))

        #match b-quarks to gen-jets by deltaR
        matches = matchObjectCollection(event.b_hadrons_sorted, event.gen_jet, 0.3)

        #find jets matched to b-quarks
        event.matched_b_jets = []
        for p in event.b_hadrons_sorted:
            if matches.has_key(p) and matches[p] != None:
                event.matched_b_jets += [matches[p]]
                self.logger.info("quark pt={0}, id={1} matched to jet pt={2}, id={3}".format(
                    p.p4().Pt(), p.pdg_id(), matches[p].p4().Pt(), matches[p].pdg_id())
                )
            else:
                self.logger.info(
                    "b-quark with pt={0:.2f} not matched to gen-jet".format(p.p4().Pt())
                )

        #match light jets to quarks from W
        matches = matchObjectCollection(event.gen_q_w, event.gen_jet, 0.3)
        event.matched_q_jets = []
        for p in event.gen_q_w:
            if matches.has_key(p) and matches[p] != None:
                event.matched_q_jets += [matches[p]]
                self.logger.info("light quark pt={0}, id={1} matched to jet pt={2}, id={3}".format(
                    p.p4().Pt(), p.pdg_id(), matches[p].p4().Pt(), matches[p].pdg_id())
                )


        event.repr_matched_jets = EventRepresentation(
            b_quarks = event.matched_b_jets,
            l_quarks = event.matched_q_jets,
            leptons = event.gen_lep,
            invisible = event.gen_met,
        )

        event.interpretations["gen_b_jet"] = event.repr_matched_jets.make_interpretation("0w2h2t", self.conf)
        event.interpretations["gen_b_q_jet"] = event.repr_matched_jets.make_interpretation("2w2h2t", self.conf)

        #Variate JER
        event.repr_matched_jets_jer10 = copy.deepcopy(event.repr_matched_jets)
        event.repr_matched_jets_jer10.perturb_momentum(
            res = {"b_quarks": 0.1, "l_quarks": 0.1}
        )
        event.interpretations["gen_b_q_jet_jer10"] = event.repr_matched_jets_jer10.make_interpretation("2w2h2t", self.conf)

        #Variate jet scale
        event.repr_matched_jets_jes10 = copy.deepcopy(event.repr_matched_jets)
        event.repr_matched_jets_jes10.perturb_momentum(
            scale = {"b_quarks": 0.1, "l_quarks": 0.1}
        )
        event.interpretations["gen_b_q_jet_jes10"] = event.repr_matched_jets_jes10.make_interpretation("2w2h2t", self.conf)


        event.repr_matched_jets_flavour = copy.deepcopy(event.repr_matched_jets)
        event.repr_matched_jets_flavour.perturb_flavour(
            effs = {"b": 0.7, "l": 0.001}
        )
        event.interpretations["gen_b_q_jet_flavour"] = event.repr_matched_jets_flavour.make_interpretation("2w2h2t", self.conf)

class OutputsAnalyzer(Analyzer):
    def __init__(self, cfg_ana, cfg_comp, looperName):
        super(OutputsAnalyzer, self).__init__(cfg_ana, cfg_comp, looperName)
        self.conf = cfg_ana._conf

    def process(self, event):
        for (k, v) in event.interpretations.items():
            setattr(event, "interp_" + k, v)

if __name__ == "__main__":

    import PhysicsTools.HeppyCore.framework.config as cfg

    logging.basicConfig(level=logging.ERROR)

    event_ana = cfg.Analyzer(
        EventAnalyzer,
        'events'
    )
    genquark_ana = cfg.Analyzer(
        GenQuarkLevelAnalyzer,
        'genquark',
        _conf = conf,
    )
    genjet_ana = cfg.Analyzer(
        GenJetLevelAnalyzer,
        'genjet',
        _conf = conf,
    )  
    mem_ana = cfg.Analyzer(
        MEMAnalyzer,
        'mem',
        _conf = conf,
    )
    outs_ana = cfg.Analyzer(
        OutputsAnalyzer,
        'outs',
        _conf = conf,
    )
    treeProducer = cfg.Analyzer(
        class_object = AutoFillTreeProducer,
        verbose = False,
        vectorTree = True,
        globalVariables = [
        ],
        globalObjects = {
            "interp_gen_b_quark" : NTupleObject("I_sl_0w2h2t_q", interp_type, help="only 4 b-quarks interpretation"),
            "interp_gen_b_q_quark" : NTupleObject("I_sl_2w2h2t_q", interp_type, help="4 b-quarks, 2 light quarks interpretation"),
            "interp_gen_b_jet" : NTupleObject("I_sl_0w2h2t_j", interp_type, help="only 4 b-jet interpretation, matched to b-quarks"),
            "interp_gen_b_q_jet" : NTupleObject("I_sl_2w2h2t_j", interp_type, help="4 b-quarks, 2 light quarks interpretation"),
            "interp_gen_b_q_jet_jer10" : NTupleObject("I_sl_2w2h2t_j_jer10", interp_type, help="only 4 b-jet interpretation, matched to b-quarks"),
            "interp_gen_b_q_jet_jes10" : NTupleObject("I_sl_2w2h2t_j_jes10", interp_type, help="only 4 b-jet interpretation, matched to b-quarks"),
            "interp_gen_b_q_jet_flavour" : NTupleObject("I_sl_2w2h2t_j_flavour", interp_type, help="only 4 b-jet interpretation, matched to b-quarks"),
        },
        collections = {
            "gen_b_h" : NTupleCollection("gen_b_h", genParticleType, 4, help="generated quarks from H"),
            "gen_b_t" : NTupleCollection("gen_b_t", genParticleType, 4, help="generated quarks from top"),
            "gen_b_others_cleaned" : NTupleCollection("gen_b_others", genParticleType, 4, help="generated b-quarks from other sources"),
            "b_hadrons_sorted" : NTupleCollection("gen_b", genParticleType, 4, help="all chosen generated b-quarks"),
            "gen_lep" : NTupleCollection("gen_lep", genParticleType, 4, help="generated leptons"),
            "matched_b_jets" : NTupleCollection("gen_jets_b", genParticleType, 8, help="gen-jets matched to b-quarks"),
            "matched_q_jets" : NTupleCollection("gen_jets_q", genParticleType, 8, help="gen-jets matched to light quarks"),
        },
    )

    sequence = cfg.Sequence([
        event_ana,
        genquark_ana,
        genjet_ana,
        mem_ana,
        outs_ana,
        treeProducer
    ])

    from PhysicsTools.HeppyCore.framework.services.tfile import TFileService
    #output_service = cfg.Service(
    #    TFileService,
    #    'outputfile',
    #    name="outputfile",
    #    fname='tree.root',
    #    option='recreate'
    #)

    if os.environ.has_key("FILE_NAMES"):
        fns = os.environ["FILE_NAMES"].split()
        fns = map(getSitePrefix, fns)
        dataset = os.environ["DATASETPATH"]
        firstEvent = int(os.environ["SKIP_EVENTS"])
        nEvents = int(os.environ["MAX_EVENTS"])
    else:
        # fns = map(getSitePrefix, ["/store/user/jpata/tth/Sep29_v1/ttHTobb_M125_TuneCUETP8M2_ttHtranche3_13TeV-powheg-pythia8/Sep29_v1/160930_103104/0000/tree_1.root"])
        # dataset = "ttHTobb_M125_TuneCUETP8M2_ttHtranche3_13TeV-powheg-pythia8"
        # firstEvent = 0
        # nEvents = 1000

        fns = map(getSitePrefix, ["/store/user/jpata/tth/Sep29_v1/TTToSemilepton_TuneCUETP8M2_ttHtranche3_13TeV-powheg-pythia8/Sep29_v1/161005_125253/0000/tree_1.root"])
        dataset = "TTToSemilepton_TuneCUETP8M2_ttHtranche3_13TeV-powheg-pythia8"
        firstEvent = 0
        nEvents = 10000

    config = cfg.Config(
        #Run across these inputs
        components = [cfg.Component(
            dataset,
            files = fns,
            tree_name = "vhbb/tree"
        )],
        sequence = sequence,
        services = [],
        #services = [output_service],
        events_class = Events,
    )
    from PhysicsTools.HeppyCore.framework.looper import Looper
    looper = Looper(
        'Loop',
        config,
        nPrint = 0,
        firstEvent = firstEvent,
        nEvents = nEvents,
        memCheckFromEvent = True
    )
    looper.loop()
    looper.write()
