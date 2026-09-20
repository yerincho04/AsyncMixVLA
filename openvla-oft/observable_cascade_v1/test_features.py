import json
import hashlib
import tempfile
from pathlib import Path
import unittest
import numpy as np
from .features import CandidateFeatures, FEATURE_NAMES, SCHEMA
from .trigger import utility_score, ObservableCascadeTrigger
from visual_trigger_v1.trigger import VisualGateTrigger

GATE=Path(__file__).resolve().parents[1]/'visual_trigger_v1/models/visual_gate_fpr10.json'

class FeaturesTest(unittest.TestCase):
    def test_original_gate_equivalence_and_causality(self):
        a=CandidateFeatures(GATE);b=CandidateFeatures(GATE);original=VisualGateTrigger(GATE)
        rng=np.random.default_rng(13)
        rows=[]
        pooled=[]
        for i in range(7):
            obs=dict(full_image=rng.integers(0,256,(64,64,3),dtype=np.uint8),
                     wrist_image=rng.integers(0,256,(64,64,3),dtype=np.uint8),
                     state=rng.normal(size=8).astype(np.float32))
            kw=dict(observation=obs,adapter_action=rng.normal(size=7).astype(np.float32),
                    chunk_position=i%8,episode_step=i)
            rows.append(kw)
            ca,xa,sa=a.update(**kw)
            cb,xb,sb=b.update(**kw)
            ref=original.update(**kw)
            self.assertEqual(sa,ref.trigger_score)
            self.assertEqual(ca,ref.should_switch)
            np.testing.assert_array_equal(xa,xb)
            self.assertEqual(len(xa),len(FEATURE_NAMES))
            current=xa[235:747].reshape(128,2,2).mean(axis=(1,2))
            expected=current-pooled[max(0,i-4)] if i else np.zeros(128)
            np.testing.assert_array_equal(xa[747:875],expected)
            pooled.append(current.copy())
        # Mutating an old caller-owned image/state cannot alter stored history.
        rows[-1]['observation']['full_image'][:]=0
        rows[-1]['observation']['state'][:]=1000
        obs=dict(full_image=np.ones((64,64,3),np.uint8),wrist_image=np.ones((64,64,3),np.uint8),state=np.zeros(8))
        kw=dict(observation=obs,adapter_action=np.zeros(7),chunk_position=7,episode_step=7)
        np.testing.assert_array_equal(a.update(**kw)[1],b.update(**kw)[1])

    def test_rejects_privileged_channels_and_misalignment(self):
        f=CandidateFeatures(GATE)
        obs=dict(full_image=np.zeros((64,64,3),np.uint8),wrist_image=np.zeros((64,64,3),np.uint8),state=np.zeros(8))
        for name in ('env','task_id','body_xpos','onset','final_success','manifest'):
            with self.assertRaises(ValueError):
                f.update(observation={**obs,name:0},adapter_action=np.zeros(7),chunk_position=0,episode_step=0)
        with self.assertRaises(ValueError):
            f.update(observation=obs,adapter_action=np.zeros(7),chunk_position=0,episode_step=20)

    def test_serving_accepts_or_rejects_once_using_only_observable_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            gate=json.loads(GATE.read_text());gate['threshold']=1e-12
            gp=p/'gate.json';gp.write_text(json.dumps(gate))
            n=len(FEATURE_NAMES)
            art=dict(schema=SCHEMA,feature_names=list(FEATURE_NAMES),privileged_inputs=False,
                     policy='utility',threshold=0.,mean=[0.]*n,scale=[1.]*n,
                     v1_artifact=str(gp),v1_artifact_sha256=hashlib.sha256(gp.read_bytes()).hexdigest(),
                     v1_weights_sha256=hashlib.sha256(Path(gate['checkpoint']).read_bytes()).hexdigest(),
                     rescue=dict(coef=[0.]*n,intercept=2.),harm=dict(coef=[0.]*n,intercept=-2.))
            obs=dict(full_image=np.zeros((64,64,3),np.uint8),wrist_image=np.zeros((64,64,3),np.uint8),state=np.zeros(8))
            ap=p/'v2.json'
            for accept in (True,False):
                if not accept:art['rescue'],art['harm']=art['harm'],art['rescue']
                ap.write_text(json.dumps(art))
                model=ObservableCascadeTrigger(ap)
                result=model.update(observation=obs,adapter_action=np.zeros(7),chunk_position=0,episode_step=0)
                self.assertEqual(result.should_switch,accept)
                self.assertTrue(model.decided)
                self.assertFalse(model.update(observation=obs,adapter_action=np.zeros(7),chunk_position=1,episode_step=1).should_switch)
                with self.assertRaises(TypeError):
                    model.update(observation=obs,adapter_action=np.zeros(7),chunk_position=1,episode_step=1,task_id=1)

    def test_signed_utility(self):
        n=len(FEATURE_NAMES)
        art=dict(policy='utility',mean=np.zeros(n),scale=np.ones(n),
                 rescue=dict(coef=np.zeros(n),intercept=np.log(3)),
                 harm=dict(coef=np.zeros(n),intercept=-np.log(3)))
        self.assertAlmostEqual(utility_score(np.zeros(n),art),.5)
        art['rescue'],art['harm']=art['harm'],art['rescue']
        self.assertAlmostEqual(utility_score(np.zeros(n),art),-.5)

if __name__=='__main__':unittest.main()
