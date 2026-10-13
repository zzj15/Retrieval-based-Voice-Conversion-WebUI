import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch


class BundledPyMSSTests(unittest.TestCase):
    def test_imports_and_resources_ignore_installed_namesakes(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name in ("pymss", "pymss_core"):
                Path(tmp, name + ".py").write_text("raise AssertionError('External PyMSS imported')\n")
            env = dict(os.environ)
            env["PYTHONPATH"] = tmp + os.pathsep + str(Path(__file__).resolve().parents[1])
            result = subprocess.run([sys.executable, "-c", """
import sys
before = list(sys.path)
from tools.pymss import MSSeparator
from tools.pymss.model_registry import get_model_entry
from tools.pymss.modules.vocal_remover.vr_separator import VR_PARAMS_DIR
from tools.pymss.modules.bs_roformer import BSRoformer
from tools.pymss_core.modules.bs_roformer import BSRoformer as CoreBSRoformer
assert BSRoformer is CoreBSRoformer
assert get_model_entry('model_bs_roformer_ep_368_sdr_12.9628.ckpt').model_type == 'bs_roformer'
assert (VR_PARAMS_DIR / '4band_v2.json').is_file()
assert sys.path == before
assert 'pymss' not in sys.modules and 'pymss_core' not in sys.modules
"""], env=env, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_webui_model_families_synthesize_on_cpu(self):
        from tools.pymss_core.modules.bs_roformer import BSRoformer, MelBandRoformer

        torch.set_num_threads(1)
        shared = dict(dim=8, depth=1, stereo=True, heads=2, dim_head=4,
                      time_transformer_depth=1, freq_transformer_depth=1,
                      stft_n_fft=64, stft_win_length=64, stft_hop_length=16,
                      attn_dropout=0, ff_dropout=0, flash_attn=False)
        models = [BSRoformer(**shared, freqs_per_bands=(11, 11, 11)),
                  MelBandRoformer(**shared, sample_rate=16000, num_bands=4)]
        for model in models:
            with self.subTest(model=type(model).__name__), torch.no_grad():
                output = model.eval()(torch.randn(1, 2, 256))
                self.assertEqual(tuple(output.shape), (1, 2, 256))
                self.assertTrue(torch.isfinite(output).all())


if __name__ == "__main__":
    unittest.main()
