import ROOT
ROOT.gROOT.SetBatch(True)
ROOT.gSystem.Load("libFWCoreFWLite.so")
import sys, os, pickle

from PhysicsTools.HeppyCore.framework.chain import Chain as Events
from TTH.MEAnalysis.MEMUtils import set_integration_vars, add_obj
from PhysicsTools.HeppyCore.framework.analyzer import Analyzer
from PhysicsTools.Heppy.analyzers.core.AutoFillTreeProducer import *
from TTH.GenLevel.genLevelAnalysis import MEMAnalyzer, EventInterpretation, interp_type, Conf, genParticleType
from PhysicsTools.HeppyCore.utils.deltar import matchObjectCollection

from PhysicsTools.HeppyCore.framework.analyzer import Analyzer
from TTH.MEAnalysis.VHbbTree import *

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
    def from_obj(obj):
        p4 = ROOT.TLorentzVector()
        p4.SetPtEtaPhiM(obj.pt, obj.eta, obj.phi, obj.mass)
        return Particle(
            _p4=p4,
            _pdg_id = obj.pdgId,
            _status = obj.status
        )

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.physObj = self

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

class GenQuarkLevelAnalyzer(Analyzer):
    def __init__(self, cfg_ana, cfg_comp, looperName):
        super(GenQuarkLevelAnalyzer, self).__init__(cfg_ana, cfg_comp, looperName)
        self.conf = cfg_ana._conf

    def process(self, event):
        event.gen_b_h = map(lambda p: Particle.from_obj(p), event.GenBQuarkFromH)
        event.gen_b_t = map(lambda p: Particle.from_obj(p), event.GenBQuarkFromTop)
        event.gen_q_w = map(lambda p: Particle.from_obj(p), event.GenWZQuark)
        event.gen_lep = map(lambda p: Particle.from_obj(p), event.GenLepFromTop)
        event.gen_nu = map(lambda p: Particle.from_obj(p), event.GenNuFromTop)
        event.met = sum([p.p4() for p in event.gen_nu], ROOT.TLorentzVector())
        print "nGenBH={0} nGenBt={1} nGenLep={2}".format(len(event.gen_b_h), len(event.gen_b_t), len(event.gen_lep))
        event.interpretations = {}

        interp = EventInterpretation(
            b_quarks = event.gen_b_h + event.gen_b_t,
            leptons = event.gen_lep,
            invisible = event.met,
            hypo = "0w2h2t"
        )
        interp.mem_cfg = MEMAnalyzer.setup_cfg_default(self.conf)
        #disable transfer functions for the gen particles
        interp.mem_cfg.cfg.int_code -= ROOT.MEM.IntegrandType.Transfer
        #event.interpretations["gen_b_quark"] = interp

        interp = EventInterpretation(
            b_quarks = event.gen_b_h + event.gen_b_t,
            l_quarks = event.gen_q_w,
            leptons = event.gen_lep,
            invisible = event.met,
            hypo = "2w2h2t"
        )
        interp.mem_cfg = MEMAnalyzer.setup_cfg_default(self.conf)
        #disable transfer functions for the gen particles
        interp.mem_cfg.cfg.int_code -= ROOT.MEM.IntegrandType.Transfer
        event.interpretations["gen_b_q_quark"] = interp

class GenJetLevelAnalyzer(Analyzer):
    def __init__(self, cfg_ana, cfg_comp, looperName):
        super(GenJetLevelAnalyzer, self).__init__(cfg_ana, cfg_comp, looperName)
        self.conf = cfg_ana._conf

    def process(self, event):
        event.gen_jet = map(lambda p: Particle.from_obj(p), event.GenJet)
        event.gen_b = event.gen_b_h+event.gen_b_t
        matches = matchObjectCollection(event.gen_b, event.gen_jet, 0.3)
        event.matched_b_jets = []
        for p in event.gen_b:
            if matches.has_key(p) and matches[p] != None:
                event.matched_b_jets += [matches[p]]
                print "quark pt={0}, id={1} matched to jet pt={2}, id={3}".format(
                    p.p4().Pt(), p.pdg_id(), matches[p].p4().Pt(), matches[p].pdg_id())
            else:
                print "{0} not matched".format(p.p4().Pt())

        interp = EventInterpretation(
            b_quarks = event.matched_b_jets,
            leptons = event.gen_lep,
            invisible = event.met,
            hypo = "0w2h2t"
        )
        interp.mem_cfg = MEMAnalyzer.setup_cfg_default(self.conf)
        #disable transfer functions for the gen particles
        interp.mem_cfg.cfg.int_code -= ROOT.MEM.IntegrandType.Transfer
        #event.interpretations["gen_b_jet"] = interp

        matches = matchObjectCollection(event.gen_q_w, event.gen_jet, 0.3)
        event.matched_q_jets = []
        for p in event.gen_q_w:
            if matches.has_key(p) and matches[p] != None:
                event.matched_q_jets += [matches[p]]

        interp = EventInterpretation(
            b_quarks = event.matched_b_jets,
            l_quarks = event.matched_q_jets,
            leptons = event.gen_lep,
            invisible = event.met,
            hypo = "2w2h2t"
        )
        interp.mem_cfg = MEMAnalyzer.setup_cfg_default(self.conf)
        #disable transfer functions for the gen particles
        interp.mem_cfg.cfg.int_code -= ROOT.MEM.IntegrandType.Transfer
        event.interpretations["gen_b_q_jet"] = interp

class MiscAnalyzer(Analyzer):
    def __init__(self, cfg_ana, cfg_comp, looperName):
        super(MiscAnalyzer, self).__init__(cfg_ana, cfg_comp, looperName)
        self.conf = cfg_ana._conf

    def process(self, event):
        for (k, v) in event.interpretations.items():
            setattr(event, "interp_" + k, v)

if __name__ == "__main__":

    import PhysicsTools.HeppyCore.framework.config as cfg

    event_ana = cfg.Analyzer(
        EventAnalyzer,
        'events'
    )
    genquark_ana = cfg.Analyzer(
        GenQuarkLevelAnalyzer,
        'genquark',
        _conf = Conf,
    )
    genjet_ana = cfg.Analyzer(
        GenJetLevelAnalyzer,
        'genjet',
        _conf = Conf,
    )  
    mem_ana = cfg.Analyzer(
        MEMAnalyzer,
        'mem',
        _conf = Conf,
    )
    misc_ana = cfg.Analyzer(
        MiscAnalyzer,
        'misc',
        _conf = Conf,
    )
    treeProducer = cfg.Analyzer(
        class_object = AutoFillTreeProducer,
        verbose = False,
        vectorTree = True,
        globalVariables = [
        ],
        globalObjects = {
            #"interp_gen_b_quark" : NTupleObject("interp_gen_b_quark", interp_type, help="only 4 b-quarks interpretation"),
            "interp_gen_b_q_quark" : NTupleObject("interp_gen_b_q_quark", interp_type, help="4 b-quarks, 2 light quarks interpretation"),
            "interp_gen_b_q_jet" : NTupleObject("interp_gen_b_q_jet", interp_type, help="4 b-quarks, 2 light quarks interpretation"),
            #"interp_gen_b_jet" : NTupleObject("interp_gen_b_jet", interp_type, help="only 4 b-jet interpretation, matched to b-quarks"),
        },
        collections = {
            "gen_b_h" : NTupleCollection("gen_b_h", genParticleType, 3, help="generated quarks from H"),
            "gen_b_t" : NTupleCollection("gen_b_t", genParticleType, 3, help="generated quarks from top"),
        },
    )

    sequence = cfg.Sequence([
        event_ana,
        genquark_ana,
        genjet_ana,
        mem_ana,
        misc_ana,
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

    fns = os.environ["FILE_NAMES"].split()
    dataset = os.environ["DATASETPATH"]
    firstEvent = int(os.environ["SKIP_EVENTS"])
    nEvents = int(os.environ["MAX_EVENTS"])
    
    config = cfg.Config(
        #Run across these inputs
        components = [cfg.Component(
            dataset,
            files = fns,
        )],
        sequence = sequence,
        services = [output_service],
        events_class = Events,
    )
    from PhysicsTools.HeppyCore.framework.looper import Looper
    looper = Looper(
        'Loop',
        config,
        nPrint = 0,
        firstEvent = firstEvent,
        nEvents = nEvents
    )
    looper.loop()
    looper.write()
