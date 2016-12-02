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
from TTH.GenLevel.genLevelAnalysis import MEMAnalyzer, EventInterpretation, interp_type, conf, genParticleType, genJetType, fillCoreVariables
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
        return True

class Particle:
    @staticmethod
    def from_obj(obj, parent_id=0):
        p4 = ROOT.TLorentzVector()
        p4.SetPtEtaPhiM(obj.pt, obj.eta, obj.phi, obj.mass)
        return Particle(
            _p4 = p4,
            _pdg_id = obj.pdgId,
            _status = obj.status,
            numBHadrons = getattr(obj, "numBHadrons", -99),
            numCHadrons = getattr(obj, "numCHadrons", -99),
            _parent_id = parent_id
        )

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        #self.physObj = self

    def is_final_state(self):
        return True

    def pdg_id(self):
        return self._pdg_id

    def p4(self):
        return self._p4

    def set_p4(self, new_p4):
        self._p4 = new_p4

    def rescale_pt(self, pt_new):
        pt_old = self.p4().Pt()
        m_old = self.p4().M()
        p4_new = ROOT.TLorentzVector()
        p4_new.SetPtEtaPhiM(pt_new, self.p4().Eta(), self.p4().Phi(), pt_new/pt_old * m_old)
        self.set_p4(p4_new)

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
    """This summarizes the reco content of an event, which may be 'perturbed' under
    systematics.
    
    Attributes:
        leptons (list of Particle): Identified leptons in the event
        b_quarks (list of Particle): Identified b-quarks (or jets) in the event
        l_quarks (list of Particle): Identified light quarks (or jets) in the event
        invisible (TLorentzVector): MET of the event
    """
    def __init__(self, **kwargs):
        self.b_quarks = kwargs.get("b_quarks", [])
        self.l_quarks = kwargs.get("l_quarks", [])
        self.leptons = kwargs.get("leptons", [])
        self.invisible = kwargs.get("invisible", ROOT.TLorentzVector())


    @staticmethod
    def make_jet_tf(pt, res=0.1):
         """Returns a gaussian PDF for the jet momentum f(ptreco | ptgen) with
         x: reco-pt
         par 0: gen-pt
         par 1: multiplicative resolution
         
         Args:
             pt (float): jet momentum
             res (float, optional): Jet resolution as a fraction of momentum
         
         Returns:
             TF1: transfer function
         """
         f = ROOT.TF1("jet_tf", "1.0/([0]*[1]*sqrt(2*3.14159264))*exp(-0.5*TMath::Power(((x-[0])/([0]*[1])), 2))", 0, 10000)
         f.SetParameter(0, pt)
         f.SetParameter(1, res)
         return f

    @staticmethod
    def make_met_tf(pt_x, pt_y, res=0.3):
        f = ROOT.TF2("met_tf", "1.0/([0]*[1]*sqrt(2*3.14159264))*exp(-0.5*TMath::Power(((x-[2])/[0]), 2))*exp(-0.5*TMath::Power(((x-[3])/[1]), 2))", 0, 10000, 0, 10000)
        f.SetParameter(0, res*pt_x)
        f.SetParameter(1, res*pt_y)
        f.SetParameter(2, pt_x)
        f.SetParameter(3, pt_y)
        return f


    def smear_quarks(self):
        for q in self.b_quarks + self.l_quarks:
            f = self.make_jet_tf(q.p4().Pt())
            pt_new = f.GetRandom()
            logging.debug("rescaling pt {0} -> {1}".format(q.p4().Pt(), pt_new))
            q.rescale_pt(pt_new)
            q.tf = f

    def perturb_momentum(self, scale={}, res={}):
        """Changes object momentum of objects in both scale (deterministic multiplicative) and resolution (random multiplicative)
        
        res <- from normal distribution with mean 0, given sigma
        pt <- (1.0 + scale + res) * pt

        Args:
            scale (dict, optional): Dictionary with per-object scale factors
            res (dict, optional): Dictionary with per-object resolution factors
        
        Returns:
            nothing
        """

        #all objects to consider
        ks = ["b_quarks", "l_quarks", "leptons"]
        for k in ks:
            sf_scale = scale.get(k, 0.0)
            sig_res = res.get(k, 0.0)

            #loop over objects
            for iobj, obj in enumerate(getattr(self, k)):
                v_res = 0.0

                if sig_res > 0:
                    v_res = np.random.normal(0.0, sig_res)

                pt_new = (1.0 + sf_scale + v_res) * obj.p4().Pt()
                m_new = (1.0 + sf_scale + v_res) * obj.p4().M()
                obj.p4().SetPtEtaPhiM(pt_new, obj.p4().Eta(), obj.p4().Phi(), m_new)

    def perturb_flavour(self, effs={}):
        """Changes the jet-to-quark association according to b-tagging efficiencies in a random way
        
        Args:
            effs (dict, optional): Dictionary of per-flavour tagging efficiencies
        
        Returns:
            nothing
        """
        b_quarks = []
        l_quarks = []

        tag_cands = []
        for q in self.b_quarks:
            tag_cands += [(q, "b")]

        for q in self.l_quarks:
            tag_cands += [(q, "l")]

        for q, fl in tag_cands:
            #Checks if this jet would be b-tagged
            is_tagged = np.random.uniform() < effs[fl]
            if is_tagged:
                b_quarks += [q]
            else:
                l_quarks += [q]

        self.b_quarks = b_quarks
        self.l_quarks = l_quarks

    def make_interpretation(self, hypo, conf, selection_func):
        """Returns a MEM interpretation of the event, given a hypothesis.
        
        Args:
            hypo (string): MEM hypothesis
            conf (MEMConfig): MEM configuration
        
        Returns:
            TYPE: Description
        """
        if hypo == "0w2h2t":
            interp = EventInterpretation(
                b_quarks = self.b_quarks,
                leptons = self.leptons,
                invisible = self.invisible,
                hypo = "0w2h2t",
                selection_function = selection_func
            )
        elif hypo in ["1w2h2t", "2w2h2t"]:
            interp = EventInterpretation(
                b_quarks = self.b_quarks,
                l_quarks = self.l_quarks,
                leptons = self.leptons,
                invisible = self.invisible,
                hypo = hypo,
                selection_function = selection_func
            )

        #interp.mem_cfg = MEMAnalyzer.setup_cfg_default(conf)
        return interp

class GenQuarkLevelAnalyzer(Analyzer):
    """Processes the generated quarks in the event and creates a quark-level hypothesis 
    
    Attributes:
        conf (TYPE): Description
        logger (TYPE): Description
    """
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
        event.gen_lep = filter(
            lambda x, conf=self.conf: x.p4().Pt() > conf["leptons"]["pt"] and abs(x.p4().Eta()) < conf["leptons"]["eta"],
            event.gen_lep
        )
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
        for q in event.gen_b_h:
            q.source = "h"
        for q in event.gen_b_t:
            q.source = "t"
        for q in event.gen_q_w:
            q.source = "w"
        b_hadrons = event.gen_b_h + event.gen_b_t
        self.logger.debug("b from h: {0}".format(map(str, event.gen_b_h)))
        self.logger.debug("b from t: {0}".format(map(str, event.gen_b_t)))

        #Find b-hadrons in additional set that are not matched to existing higgs or top
        #derived b-hadrons
        additional_b_hadrons = []
        for bhad in event.gen_b_others:
            matches = matchObjectCollection(event.gen_b_others, b_hadrons, 0.3)
            if not matches.has_key(bhad) or matches[bhad] is None:
                self.logger.debug("no match for pt={0}, adding".format(bhad.p4().Pt()))
                bhad.source = "other"
                b_hadrons += [bhad]
                additional_b_hadrons += [bhad]
        self.logger.debug("b from other: {0}".format(map(str, additional_b_hadrons)))

        event.gen_b_others_cleaned = additional_b_hadrons

        #Apply kinematic cuts and sort by pt
        event.b_hadrons_sorted = sorted(b_hadrons, key=lambda x: x.p4().Pt(), reverse=True)
        event.b_hadrons_sorted = filter(lambda x: x.p4().Pt() > self.conf["jets"]["pt"], event.b_hadrons_sorted)
        event.b_hadrons_sorted = filter(lambda x: abs(x.p4().Eta()) < self.conf["jets"]["eta"], event.b_hadrons_sorted)
        event.b_hadrons_sorted = event.b_hadrons_sorted

        #create a quark-level representation
        event.repr_quarks = EventRepresentation(
            b_quarks = event.b_hadrons_sorted[:4],
            l_quarks = event.gen_q_w[:2],
            leptons = event.gen_lep,
            invisible = event.gen_met,
        )

        event.repr_quarks_smeared = copy.deepcopy(event.repr_quarks)
        event.repr_quarks_smeared.smear_quarks()


        event.interpretations["gen_quark_2w2h2t"] = event.repr_quarks.make_interpretation("2w2h2t", self.conf, "sl_2w2h2t")
        event.interpretations["gen_quark_2w2h2t_smeared"] = event.repr_quarks_smeared.make_interpretation("2w2h2t", self.conf, "sl_2w2h2t")

        #create the 022 and 222 interpretations on the quark level
        #event.interpretations["gen_b_quark"] = event.repr_quarks.make_interpretation("0w2h2t", self.conf, "sl_0w2h2t")
        #event.interpretations["gen_b_q_quark"] = event.repr_quarks.make_interpretation("2w2h2t", self.conf, "sl_2w2h2t")
        return True

class GenJetLevelAnalyzer(Analyzer):
    """Processes the gen-jets in the event, matching them to quarks and
    creating jet-level event interpretations
    
    Attributes:
        conf (TYPE): Description
        logger (TYPE): Description
    """
    def __init__(self, cfg_ana, cfg_comp, looperName):
        super(GenJetLevelAnalyzer, self).__init__(cfg_ana, cfg_comp, looperName)
        self.conf = cfg_ana._conf
        self.logger = logging.getLogger("GenJetLevelAnalyzer")

    def process(self, event):
        #get all the gen-jet
        event.gen_jet = map(lambda p: Particle.from_obj(p, 0), event.GenJet)
        event.gen_jet = filter(
            lambda x, conf=conf: x.p4().Pt() > conf["jets"]["pt"] and abs(x.p4().Eta()) < conf["jets"]["eta"],
            event.gen_jet
        )

        #match b-quarks to gen-jets by deltaR
        matches = matchObjectCollection(event.b_hadrons_sorted, event.gen_jet, 0.3)

        #find jets matched to b-quarks
        event.matched_b_jets = []
        for p in event.b_hadrons_sorted:
            if matches.has_key(p) and matches[p] != None:
                jet = matches[p]
                jet.source = p.source
                event.matched_b_jets += [jet]
                self.logger.debug("quark pt={0}, id={1} (from {2}) matched to jet pt={3}, id={4}".format(
                    p.p4().Pt(), p.pdg_id(), p.source, matches[p].p4().Pt(), matches[p].pdg_id())
                )
            else:
                self.logger.debug(
                    "b-quark with pt={0:.2f} not matched to gen-jet".format(p.p4().Pt())
                )

        #find jets matched to light quarks
        matches = matchObjectCollection(event.gen_q_w, event.gen_jet, 0.3)
        event.matched_q_jets = []
        for p in event.gen_q_w:
            if matches.has_key(p) and matches[p] != None:
                jet = matches[p]
                jet.source = p.source
                event.matched_q_jets += [jet]
                self.logger.debug("light quark pt={0}, id={1} matched to jet pt={2}, id={3}".format(
                    p.p4().Pt(), p.pdg_id(), matches[p].p4().Pt(), matches[p].pdg_id())
                )

        #find unmatched jets
        event.unmatched_jets = []
        for jet in event.gen_jet:
            if jet not in event.matched_b_jets and jet not in event.matched_q_jets:
                event.unmatched_jets += [jet]

        #count number of jets matched to quarks from certain origin
        match_count = {
            "t": 0,
            "h": 0,
            "w": 0,
            "other": 0,
        }
        for jet in event.matched_b_jets + event.matched_q_jets:
            match_count[jet.source] += 1
        event.match_count = match_count
        self.logger.info("process: matched_b_jets={0} matched_q_jets={0}".format(len(event.matched_b_jets), len(event.matched_q_jets)))

        #Creates a jet-level representation
        event.repr_matched_jets = EventRepresentation(
            b_quarks = event.matched_b_jets[:4],
            l_quarks = event.matched_q_jets[:2],
            leptons = event.gen_lep,
            invisible = event.gen_met,
        )


        event.repr_matched_jets_smeared = copy.deepcopy(event.repr_quarks)
        event.repr_matched_jets_smeared.smear_quarks()

        event.interpretations["gen_jet_matched_2w2h2t"] = event.repr_matched_jets.make_interpretation("2w2h2t", self.conf, "sl_2w2h2t")
        event.interpretations["gen_jet_matched_2w2h2t_smeared"] = event.repr_matched_jets_smeared.make_interpretation("2w2h2t", self.conf, "sl_2w2h2t")

        #event.interpretations["gen_b_q_jet_flavour"] = event.repr_matched_jets_flavour.make_interpretation("2w2h2t", self.conf, "sl_2w2h2t")
        return True

class OutputsAnalyzer(Analyzer):
    def __init__(self, cfg_ana, cfg_comp, looperName):
        super(OutputsAnalyzer, self).__init__(cfg_ana, cfg_comp, looperName)
        self.conf = cfg_ana._conf

    def process(self, event):
        for (k, v) in event.interpretations.items():
            setattr(event, "interp_" + k, v)
        return True

match_type = NTupleObjectType("match_type", variables = [
    NTupleVariable("top", lambda x : x["t"], the_type=int),
    NTupleVariable("higgs", lambda x : x["h"], the_type=int),
    NTupleVariable("w", lambda x : x["w"], the_type=int),
    NTupleVariable("other", lambda x : x["other"], the_type=int),
])

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

    AutoFillTreeProducer.fillCoreVariables = fillCoreVariables
    treeProducer = cfg.Analyzer(
        class_object = AutoFillTreeProducer,
        defaultFloatType = "F",
        verbose = True,
        vectorTree = True,
        globalVariables = [
            NTupleVariable("ttCls",  lambda ev: getattr(ev, "ttCls", -1), float,mcOnly=True, help="ttbar classification via GenHFHadronMatcher"),
        ],
        globalObjects = {
            #"interp_gen_b_quark" : NTupleObject("I_sl_0w2h2t_q", interp_type, help="only 4 b-quarks interpretation"),
            #"interp_gen_b_quark_smeared" : NTupleObject("I_sl_0w2h2t_q_smear", interp_type, help="only 4 b-quarks interpretation"),
            #"interp_gen_quark_2w2h2t" : NTupleObject("I_sl_2w2h2t_q", interp_type, help="4 b-quarks, 2 light quarks interpretation"),
            #"interp_gen_quark_2w2h2t_smeared" : NTupleObject("I_sl_2w2h2t_q_smeared", interp_type, help="4 b-quarks, 2 light quarks interpretation, TF-smeared"),
            # "interp_gen_b_jet" : NTupleObject("I_sl_0w2h2t_j", interp_type, help="only 4 b-jet interpretation, matched to b-quarks"),
            #"interp_gen_jet_matched_2w2h2t" : NTupleObject("I_sl_2w2h2t_j_matched", interp_type, help="4 b-quarks, 2 light quarks interpretation"),
            #"interp_gen_jet_matched_2w2h2t_smeared" : NTupleObject("I_sl_2w2h2t_j_matched_smeared", interp_type, help="4 b-quarks, 2 light quarks interpretation"),
            # "interp_gen_b_q_jet_jer10" : NTupleObject("I_sl_2w2h2t_j_jer10", interp_type, help="only 4 b-jet interpretation, matched to b-quarks"),
            # "interp_gen_b_q_jet_jes10" : NTupleObject("I_sl_2w2h2t_j_jes10", interp_type, help="only 4 b-jet interpretation, matched to b-quarks"),
            # "interp_gen_b_q_jet_flavour" : NTupleObject("I_sl_2w2h2t_j_flavour", interp_type, help="only 4 b-jet interpretation, matched to b-quarks"),
            "match_count" : NTupleObject("match", match_type, help="gen-jet matching"),
        },
        collections = {
            "gen_b_h" : NTupleCollection("gen_b_h", genParticleType, 4, help="generated quarks from H"),
            "gen_b_t" : NTupleCollection("gen_b_t", genParticleType, 4, help="generated quarks from top"),
            "gen_b_others_cleaned" : NTupleCollection("gen_b_others", genParticleType, 4, help="generated b-quarks from other sources"),
            "gen_q_w" : NTupleCollection("gen_q_w", genParticleType, 8, help="generated quarks from W/Z"),
            "b_hadrons_sorted" : NTupleCollection("gen_b", genParticleType, 4, help="all chosen generated b-quarks"),

            "gen_lep" : NTupleCollection("gen_lep", genParticleType, 4, help="generated leptons after selection"),
            "gen_jet": NTupleCollection("gen_jet", genParticleType, 20, help="generated jets after selection"),
#            "matched_b_jets" : NTupleCollection("gen_jets_b", genJetType, 8, help="gen-jets matched to b-quarks"),
#            "matched_q_jets" : NTupleCollection("gen_jets_q", genJetType, 8, help="gen-jets matched to light quarks"),
#            "unmatched_jets" : NTupleCollection("gen_jets_unmatched", genJetType, 8, help="gen-jets not matched to quarks"),
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
    output_service = cfg.Service(
        TFileService,
        'outputfile',
        fname='tree.root',
        option='recreate'
    )

    if os.environ.has_key("FILE_NAMES"):
        fns = os.environ["FILE_NAMES"].split()
        fns = map(getSitePrefix, fns)
        dataset = os.environ["DATASETPATH"]
        firstEvent = int(os.environ["SKIP_EVENTS"])
        nEvents = int(os.environ["MAX_EVENTS"])
    elif os.environ.get("SAMPLE", "tth") == "tth":
        fns = map(getSitePrefix, ["/store/user/jpata/tth/Sep29_v1/ttHTobb_M125_TuneCUETP8M2_ttHtranche3_13TeV-powheg-pythia8/Sep29_v1/160930_103104/0000/tree_1.root"])
        dataset = "ttHTobb_M125_TuneCUETP8M2_ttHtranche3_13TeV-powheg-pythia8"
        firstEvent = 0
        nEvents = 100
    elif os.environ.get("SAMPLE", "tth") == "ttjets":
        fns = map(getSitePrefix, ["/store/user/jpata/tth/Sep29_v1/TTToSemilepton_TuneCUETP8M2_ttHtranche3_13TeV-powheg-pythia8/Sep29_v1/161005_125253/0000/tree_1.root"])
        dataset = "TTToSemilepton_TuneCUETP8M2_ttHtranche3_13TeV-powheg-pythia8"
        firstEvent = 0
        nEvents = 1000

    print "files", fns
    config = cfg.Config(
        #Run across these inputs
        components = [cfg.MCComponent(
            dataset,
            files = fns,
            tree_name = "vhbb/tree",
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
        nEvents = nEvents,
    )
    looper.loop()
    looper.write()

    tf = ROOT.TFile.Open("{0}/tree.root".format(looper.name), "UPDATE")
    for fn in fns:
        inf = ROOT.TFile.Open(fn)
        hcount = inf.Get("vhbb/Count")
        tf.cd()
        if tf.Get("Count") == None:
            hcount.Clone()
        else:
            tf.Get("Count").Add(hcount)
        inf.Close()
