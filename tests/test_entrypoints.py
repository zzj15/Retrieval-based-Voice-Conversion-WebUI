import argparse
import importlib.util
import io
import json
import logging
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from configs import config as config_module
from configs.cli import parse_launch_args


ROOT = Path(__file__).resolve().parents[1]


class LaunchConfigurationTests(unittest.TestCase):
    def test_launch_flags_and_port_fallback(self):
        python_cmd = r"C:\程序 Files\Python\python.exe"
        options = parse_launch_args([
            "--port", "8765", "--pycmd", python_cmd, "--colab",
            "--noparallel", "--noautoopen", "--dml",
        ])
        self.assertEqual(vars(options), dict(
            listen_port=8765, python_cmd=python_cmd, iscolab=True,
            noparallel=True, noautoopen=True, dml=True,
        ))
        for port, expected in ((-1, 7865), (0, 0), (65535, 65535), (65536, 7865)):
            with self.subTest(port=port):
                self.assertEqual(parse_launch_args(["--port", str(port)]).listen_port, expected)
        with mock.patch("sys.stderr", new=io.StringIO()), self.assertRaises(SystemExit) as error:
            parse_launch_args(["--host-only-flag"])
        self.assertEqual(error.exception.code, 2)

    def test_config_ignores_host_argv_and_keeps_instances_independent(self):
        host_argv = ["host", "--map", "vst-map", "--port", "9999"]
        with mock.patch.object(sys, "argv", host_argv), mock.patch.object(
            config_module, "infer_device", torch.device("cpu")
        ), mock.patch.object(config_module, "infer_dtype", torch.float32):
            default = config_module.Config()
            custom = config_module.Config(**vars(parse_launch_args([
                "--port", "8765", "--pycmd", "custom python", "--noautoopen",
            ])))
            self.assertIs(sys.argv, host_argv)
            self.assertEqual(sys.argv, ["host", "--map", "vst-map", "--port", "9999"])
        self.assertIsNot(default, custom)
        self.assertEqual(default.listen_port, 7865)
        self.assertEqual(custom.listen_port, 8765)
        self.assertEqual(custom.python_cmd, "custom python")
        self.assertTrue(custom.noautoopen)
        self.assertFalse(default.noautoopen)
        self.assertEqual(default.device, "cpu")
        self.assertEqual(default.dtype, torch.float32)
        self.assertFalse(default.is_half)
        self.assertEqual((default.x_pad, default.x_query, default.x_center, default.x_max), (1, 6, 38, 41))
        custom.json_config["v1/32k.json"]["test"] = True
        self.assertNotIn("test", default.json_config["v1/32k.json"])

    def test_directml_can_be_explicit_or_automatic(self):
        device = torch.device("privateuseone:0")
        directml = SimpleNamespace(default_device=lambda: 0, device=lambda index: device)
        for infer_device, explicit in ((torch.device("cpu"), True), (device, False)):
            with self.subTest(explicit=explicit), mock.patch.dict(
                sys.modules, torch_directml=directml
            ), mock.patch.object(config_module, "infer_device", infer_device), mock.patch.object(
                config_module, "infer_dtype", torch.float32
            ):
                config = config_module.Config(dml=explicit)
                self.assertTrue(config.dml)
                self.assertEqual(config.device, device)
                self.assertEqual(config.dtype, torch.float32)
                self.assertFalse(config.is_half)


class LoggingTests(unittest.TestCase):
    def test_library_imports_preserve_host_logging_and_argv(self):
        for configured in (False, True):
            with self.subTest(configured=configured):
                result = subprocess.run([sys.executable, "-c", """
import logging, sys
root = logging.getLogger()
if sys.argv[1] == 'True':
    root.addHandler(logging.StreamHandler())
    root.setLevel(logging.ERROR)
before = (root.level, root.handlers[:])
sys.argv = ['host', '--map', 'vst-map', '--unknown-host-flag']
host_argv = sys.argv
from train import utils
from tools import pymss_webui
assert (root.level, root.handlers) == before
assert sys.argv is host_argv
assert sys.argv == ['host', '--map', 'vst-map', '--unknown-host-flag']
""", str(configured)], cwd=ROOT, capture_output=True, text=True, timeout=60)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_training_spawn_import_configures_info_console(self):
        result = subprocess.run([sys.executable, "-c", """
import logging, runpy, sys
sys.argv = ['train', '--help']
try:
    runpy.run_module('train.train', run_name='__mp_main__')
except SystemExit as error:
    assert error.code == 0
else:
    raise AssertionError('Training help did not exit')
assert logging.getLogger().level == logging.INFO
assert logging.getLogger().handlers
assert not logging.getLogger('numba').isEnabledFor(logging.DEBUG)
"""], cwd=ROOT, capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_training_file_keeps_debug_logging(self):
        from train import utils

        root = logging.getLogger()
        before = (root.level, root.handlers[:])
        old_logger = utils.logger
        with tempfile.TemporaryDirectory(prefix="rvc-log-") as temp_root:
            logger = utils.get_logger(temp_root)
            try:
                logger.debug("debug detail retained")
                self.assertIn("debug detail retained", Path(temp_root, "train.log").read_text())
                self.assertEqual((root.level, root.handlers), before)
            finally:
                for handler in logger.handlers[:]:
                    handler.close()
                    logger.removeHandler(handler)
                utils.logger = old_logger


class WorkerConfigurationTests(unittest.TestCase):
    def test_vst_lifecycle_preserves_worker_flags(self):
        module_spec = importlib.util.spec_from_file_location(
            "rvc_worker_test", ROOT / "RVCRealtimeVST/worker/rvc_worker.py"
        )
        worker = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(worker)

        class SharedBuffer(bytearray):
            closed = False

            def flush(self):
                pass

            def close(self):
                self.closed = True

        shared = SharedBuffer(worker.MAP_BYTES)
        worker.write_value(shared, 0, "I", worker.MAGIC)
        worker.write_value(shared, 4, "I", worker.PROTOCOL_VERSION)
        request_event, response_event = mock.Mock(), mock.Mock()

        def stop_worker(timeout):
            worker.write_value(shared, 8, "i", worker.STATUS_STOP)
            return worker.WAIT_OBJECT_0

        request_event.wait.side_effect = stop_worker
        engine = SimpleNamespace(effective_crossfade_ms=40, prewarm=lambda: None)
        host_argv = ["rvc_worker.py", "--map", "vst-map", "--request", "request",
                     "--response", "response", "--config", "worker.json"]

        def create_engine(cfg):
            self.assertIs(sys.argv, host_argv)
            config_module.Config()
            return engine

        with tempfile.TemporaryDirectory() as temp_root:
            config_path = Path(temp_root, "worker.json")
            config_path.write_text("{}")
            args = argparse.Namespace(map="vst-map", request="request", response="response", config=str(config_path))
            with mock.patch.object(sys, "argv", host_argv), mock.patch.object(
                worker.mmap, "mmap", return_value=shared
            ), mock.patch.object(worker, "WinEvent", side_effect=[request_event, response_event]), mock.patch.object(
                worker, "RVCStreamEngine", side_effect=create_engine
            ):
                result = worker.run(args)
                error_path = Path(args.config + ".log")
                self.assertEqual(result, 0, error_path.read_text(encoding="utf-8") if error_path.exists() else "")
                self.assertIs(sys.argv, host_argv)
        request_event.close.assert_called_once()
        response_event.close.assert_called_once()
        self.assertTrue(shared.closed)

    def test_pymss_device_choice_reaches_worker_and_separator(self):
        from tools import pymss_webui as pymss

        worker_logger = logging.getLogger("rvc.pymss.worker")
        previous_logging = (worker_logger.handlers[:], worker_logger.level, worker_logger.propagate)
        self.addCleanup(lambda: setattr(worker_logger, "handlers", previous_logging[0]))
        self.addCleanup(lambda: worker_logger.setLevel(previous_logging[1]))
        self.addCleanup(lambda: setattr(worker_logger, "propagate", previous_logging[2]))
        for device, dml in (("cpu", False), ("privateuseone:0", True)):
            with self.subTest(device=device), tempfile.TemporaryDirectory() as temp_root:
                input_path = Path(temp_root, "input.wav")
                input_path.touch()
                config = SimpleNamespace(device=device, is_half=False)
                batch = mock.MagicMock()
                batch.__enter__.return_value = batch
                batch.model_dtype = "float32"
                batch.separate_file.return_value = dict(inference_seconds=0.1, encode_seconds=0.1)
                emitted = []

                def launch_worker(command, **kwargs):
                    request = json.loads(Path(command[-1]).read_text(encoding="utf-8"))
                    self.assertEqual(request["dml"], dml)
                    self.assertEqual(pymss._pymss_worker_main(command[-1]), 0)
                    stdout = io.StringIO("".join(json.dumps(event) + "\n" for event in emitted))
                    return SimpleNamespace(stdout=stdout, poll=lambda: 0, wait=lambda: 0)

                with mock.patch.object(pymss, "Config", return_value=config) as config_factory, mock.patch.object(
                    pymss, "MSSTBatchSeparator", return_value=batch
                ) as separator, mock.patch.object(pymss, "_worker_emit", side_effect=emitted.append), mock.patch.object(
                    pymss, "_dml_model_dtype_attempts", return_value=("float32",)
                ), mock.patch.object(pymss.subprocess, "Popen", side_effect=launch_worker):
                    list(pymss.pymss_separate(
                        pymss.MODEL_SPECS[0].model_id, temp_root, temp_root, [], temp_root, "wav", config=config,
                    ))
                    config_factory.assert_called_once_with(dml=dml)
                    separator.assert_called_once()
                    self.assertIs(separator.call_args.kwargs["config"], config)
                self.assertEqual(emitted[-1]["event"], "done")
                self.assertEqual(emitted[-1]["successful"], 1)


if __name__ == "__main__":
    unittest.main()
