import gc
import tempfile
import unittest
import weakref
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from infer.checkpoint import load_synthesizer, normalize_speaker_info
from infer.cli import load_model_metadata, select_speaker
from infer.module import models


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def checkpoint(self, version="v2", f0=1):
        cls = getattr(models, "SynthesizerTrnMs%sNSFsid%s" % (
            256 if version == "v1" else 768, "" if f0 else "_nono"
        ))
        config = [17, 4, 4, 4, 8, 2, 1, 3, 0, "1", [3], [[1, 3, 5]],
                  [2, 2], 32, [4, 4], 2, 4, 16000]
        model = cls(*config, is_half=False)
        config[-3] = 109  # Loading must derive the count from the saved embedding.
        path = Path(self.tmp.name) / (version + str(f0) + ".pth")
        torch.save({
            "config": config, "version": version, "f0": f0,
            "weight": {k: v.half() for k, v in model.state_dict().items()
                       if not k.startswith("enc_q.")},
            "speaker_info": [{"id": 1, "name": "B"}, {"id": 0, "name": "A"}],
        }, path)
        return path

    def test_all_model_variants_load_and_synthesize(self):
        for version in ("v1", "v2"):
            for f0 in (0, 1):
                with self.subTest(version=version, f0=f0):
                    model, checkpoint = load_synthesizer(self.checkpoint(version, f0))
                    self.assertEqual(checkpoint["config"][-3], 2)
                    self.assertEqual(model.emb_g.num_embeddings, 2)
                    self.assertFalse(hasattr(model, "enc_q"))
                    self.assertFalse(model.training)
                    features = torch.randn(1, 6, 256 if version == "v1" else 768)
                    args = [features, torch.tensor([6])]
                    if f0:
                        args += [torch.full((1, 6), 100), torch.full((1, 6), 220.0)]
                    args.append(torch.tensor([1]))
                    with torch.no_grad():
                        waveform = model.infer(*args)[0]
                    self.assertEqual(tuple(waveform.shape), (1, 1, 24))
                    self.assertTrue(torch.isfinite(waveform).all())

    def test_precision_and_realtime_weight_norm_option(self):
        path = self.checkpoint()
        half_model, _ = load_synthesizer(path, is_half=True)
        self.assertEqual(next(half_model.parameters()).dtype, torch.float16)
        realtime_model, _ = load_synthesizer(path, remove_weight_norm=True)
        self.assertEqual(next(realtime_model.parameters()).dtype, torch.float32)
        self.assertFalse(any(k.endswith("weight_g") for k in realtime_model.state_dict()))

    def test_metadata_and_speaker_selection(self):
        rows = [{"id": 1, "name": "B"}, {"id": "0", "name": "A"},
                {"id": 1, "name": "duplicate"}, {"id": -1, "name": "negative"},
                {"id": 109, "name": "last"}, {"id": 110, "name": "overflow"},
                {"id": 0, "name": ""}, {}, None]
        self.assertEqual(normalize_speaker_info(rows, 2),
                         [{"id": 0, "name": "A"}, {"id": 1, "name": "B"}])
        self.assertEqual([s["id"] for s in normalize_speaker_info(rows)], [0, 1, 109])
        count, speakers = load_model_metadata(self.checkpoint())
        self.assertEqual(count, 2)
        self.assertEqual(select_speaker(count, speakers, None), 0)
        with self.assertRaises(ValueError):
            select_speaker(count, speakers, 2)

    def test_unloading_releases_models_before_and_after_first_inference(self):
        from infer.vc.modules import VC

        for has_hubert in (False, True):
            with self.subTest(has_hubert=has_hubert):
                vc = VC(SimpleNamespace())
                vc.net_g = torch.nn.Linear(2, 2)
                model_ref = weakref.ref(vc.net_g)
                vc.hubert_model = torch.nn.Linear(2, 2) if has_hubert else None
                vc.pipeline = object()
                vc.cpt = {"version": "v2", "f0": 1}
                with patch("infer.vc.modules.build_synthesizer",
                           side_effect=AssertionError("Unloading must not load a model")):
                    vc.get_vc("")
                    vc.get_vc("")
                gc.collect()
                self.assertIsNone(model_ref())
                for field in ("net_g", "hubert_model", "pipeline", "cpt",
                              "n_spk", "tgt_sr", "version", "if_f0"):
                    self.assertIsNone(getattr(vc, field))

    def test_switching_releases_previous_model_before_building_replacement(self):
        from infer.checkpoint import build_synthesizer
        from infer.vc.modules import VC

        path = self.checkpoint()
        config = SimpleNamespace(device="cpu", is_half=False,
                                 x_pad=1, x_query=1, x_center=1, x_max=1)
        vc = VC(config)
        vc.net_g = torch.nn.Linear(2, 2)
        previous = weakref.ref(vc.net_g)

        def build_replacement(*args, **kwargs):
            self.assertIsNone(previous())
            return build_synthesizer(*args, **kwargs)

        with patch.dict("os.environ", weight_root=self.tmp.name), \
                patch("infer.vc.modules.build_synthesizer", side_effect=build_replacement):
            vc.get_vc(path.name)
        self.assertEqual(vc.n_spk, 2)


if __name__ == "__main__":
    unittest.main()
