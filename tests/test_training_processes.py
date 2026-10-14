import ast
import builtins
import copy
import json
import os
import pathlib
import platform
import random
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest import mock

from tools import process_utils
from tools.file_io import read_text
from train import utils
from test_training import ROOT, task_namespace


def launch_namespace(root):
    namespace = task_namespace()
    configs = {
        key: json.loads((ROOT / "configs" / key).read_text(encoding="utf8"))
        for key in ("v1/32k.json", "v1/40k.json", "v1/48k.json", "v2/32k.json", "v2/48k.json")
    }
    namespace.update(
        os=os, platform=platform, subprocess=subprocess, time=time, sleep=time.sleep,
        pathlib=pathlib, json=json, copy=copy, shuffle=random.shuffle,
        now_dir=str(root), outside_index_root=str(root / "外部 索引"),
        config=SimpleNamespace(python_cmd=sys.executable, noparallel=True,
                               preprocess_per=3.0, n_cpu=2, device="cpu",
                               dml=False, is_half=False, json_config=configs),
        sr_dict={"40k": 40000}, is_multispeaker_mode=lambda mode: mode == "multi",
        prepare_multispeaker_manifest=mock.Mock(),
        validate_preprocess_outputs=mock.Mock(), validate_feature_outputs=mock.Mock(),
        read_log=lambda path: "", read_text=read_text, weight_names=lambda: [],
        Popen=mock.Mock(side_effect=lambda *args, **kwargs: SimpleNamespace(
            pid=123, returncode=0, poll=lambda: 0,
        )),
    )
    return namespace


def parse_script_header(command):
    path = ROOT / command[1]
    tree = ast.parse(path.read_text(encoding="utf8"))
    names = {"train/preprocess.py": "inp_root", "train/dataset/extract_f0.py": "mode",
             "train/dataset/extract_hubert_feature.py": "device", "train/train_index.py": "exp_name"}
    start = next(i for i, node in enumerate(tree.body)
                 if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
                 and node.targets[0].id == names[command[1]])
    body = []
    for node in tree.body[start:]:
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef)) or isinstance(node, ast.Expr):
            break
        body.append(node)
    namespace = dict(sys=SimpleNamespace(argv=command[1:]),
                     os=SimpleNamespace(environ={}, path=os.path))
    directml = SimpleNamespace(default_device=lambda: 0, device=lambda value: "privateuseone:0")
    with mock.patch.dict(sys.modules, torch_directml=directml):
        exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


class TrainingTextIOTests(unittest.TestCase):
    def test_utf8_filelist_survives_windows_legacy_default_encoding(self):
        value = "带空格 音频.wav|feature.npy|0\n"
        real_open = builtins.open

        def legacy_open(*args, **kwargs):
            kwargs.setdefault("encoding", "cp1252")
            return real_open(*args, **kwargs)

        with tempfile.TemporaryDirectory() as folder:
            path = pathlib.Path(folder) / "filelist.txt"
            path.write_text(value, encoding="utf8")
            with mock.patch("builtins.open", side_effect=legacy_open):
                self.assertEqual(utils.load_filepaths_and_text(path),
                                 [["带空格 音频.wav", "feature.npy", "0"]])

    def test_legacy_gbk_text_still_decodes_when_locale_is_utf8(self):
        value = "带空格 音频.wav|feature.npy|0\n"
        real_open = builtins.open

        def utf8_open(*args, **kwargs):
            kwargs.setdefault("encoding", "utf8")
            return real_open(*args, **kwargs)

        with tempfile.TemporaryDirectory() as folder:
            path = pathlib.Path(folder) / "filelist.txt"
            path.write_bytes(value.encode("gbk"))
            with mock.patch("builtins.open", side_effect=utf8_open):
                self.assertEqual(read_text(path), value)


class TrainingLaunchTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="rvc-launch-")
        self.addCleanup(self.temporary.cleanup)
        self.root = pathlib.Path(self.temporary.name)
        self.namespace = launch_namespace(self.root)
        _, self.state = self.namespace["begin_train_task"]("test task")

    def commands(self):
        return [call.args[0] for call in self.namespace["Popen"].call_args_list]

    def test_preprocess_path_is_literal_in_real_child(self):
        (self.root / "train").mkdir()
        (self.root / "train/preprocess.py").write_text(
            "import json, sys\nfrom pathlib import Path\n"
            "Path('received.json').write_text(json.dumps(sys.argv))\n"
        )
        self.namespace["Popen"] = subprocess.Popen
        self.namespace["kill_process_tree"] = process_utils.kill_process_tree
        value = str(self.root / "带 空格 $(printf changed) & %RVC_LITERAL%")
        try:
            list(self.namespace["run_preprocess_dataset"](
                value, "实验 $()", "40k", 1, self.state,
            ))
            argv = json.loads((self.root / "received.json").read_text())
            self.assertEqual(argv[1], value)
            self.assertEqual(argv[4], str(self.root / "logs" / "实验 $()"))
        finally:
            self.namespace["finish_train_task"](self.state)

    def test_preprocess_manifest_and_partition_arguments_match_backend_parser(self):
        list(self.namespace["run_preprocess_dataset"](
            "带 空格 音频", "experiment", "40k", 2, self.state, training_mode="multi",
        ))
        header = parse_script_header(self.commands()[0])
        self.assertEqual(header["inp_root"], "带 空格 音频")
        self.assertEqual((header["sr"], header["n_p"], header["noparallel"], header["per"]),
                         (40000, 2, True, 3.0))
        self.assertEqual(header["manifest_path"], str(self.root / "logs/experiment/multispeaker_manifest.json"))

    def test_cpu_cuda_and_directml_feature_arguments_match_backend_parsers(self):
        cases = (
            (False, "pm", "", "", "cpu", False),
            (False, "rmvpe", "", "", "cpu", False),
            (False, "rmvpe", "0-1", "0-1", "cuda:0", True),
            (True, "rmvpe", "", "-", "privateuseone:0", False),
        )
        for dml, method, gpus, rmvpe, device, half in cases:
            with self.subTest(dml=dml, method=method, gpus=gpus):
                namespace = launch_namespace(self.root)
                _, state = namespace["begin_train_task"]("features")
                namespace["config"].dml, namespace["config"].device = dml, device
                namespace["config"].is_half = half
                list(namespace["run_extract_f0_feature"](
                    gpus, 2, method, True, "experiment", "v2", rmvpe, state,
                ))
                commands = [call.args[0] for call in namespace["Popen"].call_args_list]
                f0 = [parse_script_header(cmd) for cmd in commands if cmd[1].endswith("extract_f0.py")]
                features = [parse_script_header(cmd) for cmd in commands if cmd[1].endswith("extract_hubert_feature.py")]
                self.assertTrue(f0 and features)
                for header in f0 + features:
                    self.assertEqual(header["exp_dir"], str(self.root / "logs/experiment"))
                if gpus:
                    self.assertEqual([h["i_gpu"] for h in f0], ["0", "1"])
                    self.assertEqual([h["i_part"] for h in f0], [0, 1])
                    self.assertEqual([h["i_gpu"] for h in features], ["0", "1"])
                    self.assertTrue(all(h["n_part"] == 2 for h in f0 + features))
                elif dml:
                    self.assertEqual(f0[0]["device"], "privateuseone:0")
                else:
                    self.assertEqual((f0[0]["n_p"], f0[0]["f0method"]), (2, method))
                self.assertTrue(all(h["version"] == "v2" for h in features))

    def test_model_flags_preserve_pretrained_paths_and_cpu_gpu_options(self):
        exp = "模型 实验"
        for folder in ("0_gt_wavs", "3_feature768", "2a_f0", "2b-f0nsf"):
            directory = self.root / "logs" / exp / folder
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "row.npy").touch()
        for gpus in ("", "0-1"):
            with self.subTest(gpus=gpus):
                self.namespace["Popen"].reset_mock()
                generator_path = str(self.root / "预训练 G 空格.pth")
                discriminator_path = str(self.root / "预训练 D 空格.pth")
                list(self.namespace["run_train_model"](
                    exp, "40k", True, 7, 2, 4, 1, "是", generator_path,
                    discriminator_path, gpus, "否", "是", "v2", self.state,
                ))
                cmd = self.commands()[0]
                config_path = self.root / "logs" / exp / "config.json"
                with mock.patch.object(sys, "argv", cmd[1:]), mock.patch.object(
                    utils, "read_text", return_value=config_path.read_text(encoding="utf8")
                ):
                    params = utils.get_hparams()
                self.assertEqual((params.name, params.pretrainG, params.pretrainD),
                                 (exp, generator_path, discriminator_path))
                self.assertEqual((params.total_epoch, params.save_every_epoch, params.train.batch_size), (4, 2, 1))
                self.assertEqual((params.if_f0, params.if_latest, params.if_cache_data_in_gpu, params.save_every_weights),
                                 (1, 1, 0, "1"))
                self.assertEqual(params.gpus, gpus or "0")
                self.assertEqual(self.namespace["Popen"].call_args.kwargs["env"]["RVC_CUDA_GRAPH"], "0")

    def test_index_arguments_preserve_output_path_and_training_mode(self):
        for mode in (None, "single", "multi"):
            with self.subTest(mode=mode):
                self.namespace["Popen"].reset_mock()
                list(self.namespace["run_train_index"]("实验 空格", "v2", self.state, training_mode=mode))
                header = parse_script_header(self.commands()[0])
                self.assertEqual((header["exp_name"], header["version"], header["n_cpu"]),
                                 ("实验 空格", "v2", 2))
                self.assertEqual(header["outside_index_root"], str(self.root / "外部 索引"))
                self.assertEqual(header["index_mode"], mode or "auto")

    def test_platform_process_groups_and_training_environment_are_preserved(self):
        for name in ("Linux", "Windows"):
            with self.subTest(platform=name), mock.patch.object(platform, "system", return_value=name), mock.patch.object(
                subprocess, "CREATE_NEW_PROCESS_GROUP", 512, create=True
            ), mock.patch.dict(os.environ, RVC_CUDA_GRAPH="1"):
                self.namespace["start_train_process"](self.state, [sys.executable, "train/train.py", "-e", "experiment"])
                options = self.namespace["Popen"].call_args.kwargs
                self.assertFalse(options["shell"])
                self.assertEqual(options["env"]["RVC_CUDA_GRAPH"], "0")
                self.assertEqual(os.environ["RVC_CUDA_GRAPH"], "1")
                if name == "Windows":
                    self.assertEqual(options["creationflags"], 512)
                    self.assertNotIn("start_new_session", options)
                else:
                    self.assertTrue(options["start_new_session"])
                    self.assertNotIn("creationflags", options)

    def test_strings_invalid_experiments_and_gpu_ids_fail_before_spawn(self):
        with self.assertRaises(TypeError):
            self.namespace["start_train_process"](self.state, "python train/train.py")
        for name in ("", " ", "..", "../outside", r"..\outside", "C:experiment"):
            with self.subTest(experiment=name), self.assertRaises(ValueError):
                list(self.namespace["run_preprocess_dataset"]("input", name, "40k", 1, self.state))
        for gpu in ("0; command", "--1", "0--1", "cuda:0"):
            with self.subTest(gpu=gpu), self.assertRaises(ValueError):
                list(self.namespace["run_extract_f0_feature"](gpu, 1, "rmvpe", True, "experiment", "v2", "", self.state))
        self.namespace["Popen"].assert_not_called()

    def test_late_spawn_after_task_closed_is_immediately_terminated(self):
        self.namespace["finish_train_task"](self.state)
        child = mock.Mock()
        child.poll.return_value = None
        self.namespace["Popen"] = mock.Mock(return_value=child)
        self.namespace["start_train_process"](self.state, [sys.executable, "train/preprocess.py"])
        self.namespace["kill_process_tree"].assert_called_once_with(child, "test task", self.namespace["logger"])


class ProcessTreeTests(unittest.TestCase):
    def test_windows_taskkill_failure_falls_back_to_terminate_and_wait(self):
        process = mock.Mock(pid=123)
        process.poll.side_effect = [None, -15]
        fake_os = SimpleNamespace(name="nt")
        with mock.patch.object(process_utils, "os", fake_os), mock.patch.object(
            process_utils.subprocess, "run", return_value=SimpleNamespace(returncode=1)
        ) as run:
            self.assertTrue(process_utils.kill_process_tree(process))
        run.assert_called_once()
        self.assertEqual(run.call_args.args[0], ["taskkill", "/t", "/f", "/pid", "123"])
        process.terminate.assert_called_once()
        process.wait.assert_called_once_with(timeout=5)

    @unittest.skipIf(os.name == "nt", "POSIX process-group regression")
    def test_descendant_ignoring_sigterm_is_killed_after_parent_exits(self):
        with tempfile.TemporaryDirectory() as root:
            ready = pathlib.Path(root) / "ready"
            child_code = (
                "import os,signal,sys,time\nfrom pathlib import Path\n"
                "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
                "Path(sys.argv[1]).write_text(str(os.getpid()))\ntime.sleep(20)\n"
            )
            parent_code = (
                "import subprocess,sys,time\nfrom pathlib import Path\n"
                "child=subprocess.Popen([sys.executable,'-c',sys.argv[1],sys.argv[2]])\n"
                "while not Path(sys.argv[2]).exists(): time.sleep(0.01)\n"
                "print(child.pid,flush=True)\ntime.sleep(20)\n"
            )
            parent = subprocess.Popen([sys.executable, "-c", parent_code, child_code, str(ready)],
                                      stdout=subprocess.PIPE, text=True, start_new_session=True)
            unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(20)"],
                                         start_new_session=True)
            child_pid = None
            try:
                child_pid = int(parent.stdout.readline())
                self.assertTrue(process_utils.kill_process_tree(parent))
                self.assertIsNotNone(parent.poll())
                self.assertIsNone(unrelated.poll())
                for _ in range(100):
                    stat = pathlib.Path(f"/proc/{child_pid}/stat")
                    if not stat.exists() or stat.read_text().split()[2] == "Z":
                        break
                    time.sleep(0.01)
                else:
                    self.fail("Descendant survived process-group termination")
            finally:
                if parent.poll() is None:
                    os.killpg(parent.pid, signal.SIGKILL)
                parent.wait(timeout=5)
                parent.stdout.close()
                if unrelated.poll() is None:
                    unrelated.kill()
                unrelated.wait(timeout=5)
                if child_pid is not None:
                    try:
                        os.kill(child_pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass


if __name__ == "__main__":
    unittest.main()
