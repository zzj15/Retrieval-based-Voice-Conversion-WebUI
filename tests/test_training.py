import ast
import logging
import os
import subprocess
import sys
import tempfile
import threading
import traceback
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch
from scipy.io import wavfile
from torch.utils.data import DataLoader

from train import data_utils
from train import spec_cache
from tools.process_utils import kill_process_tree


ROOT = Path(__file__).resolve().parents[1]
TASKS = (
    ("preprocess_dataset", "数据切分", "run_preprocess_dataset",
     (r"C:\带 空格\音频", "experiment", "40k", 2), {"training_mode": "multi"}),
    ("extract_f0_feature", "特征提取", "run_extract_f0_feature",
     ("0-1", 2, "rmvpe", True, "experiment", "v2", "0-1"), {}),
    ("train_index", "索引训练", "run_train_index",
     ("experiment", "v2"), {"training_mode": "multi"}),
)


def task_namespace(source_path=ROOT / "webui.py"):
    names = {
        "button_update", "format_status", "begin_train_task", "stop_train_task",
        "finish_train_task", "train_task_stopped", "run_train_task",
        "preprocess_dataset", "extract_f0_feature", "train_index",
        "stop_preprocess_dataset", "stop_extract_f0_feature", "stop_train_index",
        "experiment_path", "validate_experiment_name", "training_gpu_ids",
        "start_train_process", "wait_train_processes", "run_preprocess_dataset",
        "run_extract_f0_feature", "run_train_model", "run_train_index",
        "click_train", "train1key", "format_workflow_status",
    }
    tree = ast.parse(source_path.read_text(encoding="utf8"))
    functions = [node for node in tree.body
                 if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = dict(
        TRAIN_TASK=None, TRAIN_TASK_LOCK=threading.Lock(),
        i18n=lambda text: text, traceback=traceback,
        kill_process_tree=mock.Mock(), logger=logging.getLogger("training-test"),
    )
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(source_path), "exec"),
         namespace)
    return namespace


class TrainingTaskTests(unittest.TestCase):
    def test_callbacks_forward_arguments_and_stream_status(self):
        for callback, title, runner_name, args, kwargs in TASKS:
            with self.subTest(task=callback):
                namespace = task_namespace()

                def runner(*received, state, **options):
                    self.assertEqual(received, args)
                    self.assertEqual(options, kwargs)
                    self.assertIs(state, namespace["TRAIN_TASK"])
                    self.assertEqual(state["name"], title)
                    yield "first progress"
                    yield "second progress"

                namespace[runner_name] = runner
                updates = list(namespace[callback](*args, **kwargs))
                self.assertIn("正在启动", updates[0][0])
                self.assertEqual([row[0] for row in updates[1:-1]],
                                 ["first progress", "second progress"])
                self.assertIn("已完成", updates[-1][0])
                for row in updates[:-1]:
                    self.assertEqual(len(row), 3)
                    self.assertFalse(row[1]["visible"])
                    self.assertTrue(row[2]["visible"])
                self.assertTrue(updates[-1][1]["visible"])
                self.assertFalse(updates[-1][2]["visible"])
                self.assertIsNone(namespace["TRAIN_TASK"])

    def test_busy_task_keeps_owner_and_does_not_start_runner(self):
        for callback, title, runner_name, args, kwargs in TASKS:
            with self.subTest(task=callback):
                namespace = task_namespace()
                _, owner = namespace["begin_train_task"]("existing task")
                namespace[runner_name] = mock.Mock()
                updates = list(namespace[callback](*args, **kwargs))
                self.assertEqual(len(updates), 1)
                self.assertIn("等待中", updates[0][0])
                self.assertIn("existing task", updates[0][0])
                self.assertEqual(updates[0][1:],
                                 ({"__type__": "update"}, {"__type__": "update"}))
                self.assertIs(namespace["TRAIN_TASK"], owner)
                namespace[runner_name].assert_not_called()

    def test_failures_reset_buttons_and_allow_next_task(self):
        for callback, title, runner_name, args, kwargs in TASKS:
            with self.subTest(task=callback):
                namespace = task_namespace()
                namespace[runner_name] = mock.Mock(side_effect=RuntimeError("worker failed"))
                updates = list(namespace[callback](*args, **kwargs))
                self.assertIn("失败", updates[-1][0])
                self.assertIn("RuntimeError: worker failed", updates[-1][0])
                self.assertTrue(updates[-1][1]["visible"])
                self.assertFalse(updates[-1][2]["visible"])
                self.assertIsNone(namespace["TRAIN_TASK"])
                self.assertEqual(namespace["begin_train_task"]("next task")[0], "start")

    def test_stop_releases_task_and_preserves_progress(self):
        for callback, title, runner_name, args, kwargs in TASKS:
            with self.subTest(task=callback):
                namespace = task_namespace()
                process = mock.Mock()
                process.poll.return_value = None
                namespace["kill_process_tree"].side_effect = lambda *args: setattr(
                    process.poll, "return_value", -15
                )

                def runner(*received, state, **options):
                    state["processes"].append(process)
                    yield "working"
                    self.assertTrue(state["stop_requested"])

                namespace[runner_name] = runner
                updates = namespace[callback](*args, **kwargs)
                next(updates)
                self.assertEqual(next(updates)[0], "working")
                stop_name = {
                    "preprocess_dataset": "stop_preprocess_dataset",
                    "extract_f0_feature": "stop_extract_f0_feature",
                    "train_index": "stop_train_index",
                }[callback]
                self.assertIn("已停止", namespace[stop_name]()[0])
                namespace["kill_process_tree"].assert_called_once_with(
                    process, title, namespace["logger"]
                )
                self.assertIn("已停止", list(updates)[-1][0])
                self.assertIsNone(namespace["TRAIN_TASK"])

    def test_closing_callback_releases_task_at_any_yield(self):
        for callback, title, runner_name, args, kwargs in TASKS:
            for progress_count in (1, 2):
                with self.subTest(task=callback, progress=progress_count):
                    namespace = task_namespace()
                    namespace[runner_name] = lambda *args, **kwargs: iter(("working", "more"))
                    updates = namespace[callback](*args, **kwargs)
                    for _ in range(progress_count):
                        next(updates)
                    self.assertIsNotNone(namespace["TRAIN_TASK"])
                    updates.close()
                    self.assertIsNone(namespace["TRAIN_TASK"])

    def test_close_or_failure_stops_real_child_before_releasing_slot(self):
        for action in ("close", "failure"):
            with self.subTest(action=action):
                namespace = task_namespace()
                namespace["kill_process_tree"] = kill_process_tree
                children = []

                def runner(*args, state, **kwargs):
                    options = {"start_new_session": True} if os.name != "nt" else {}
                    child = subprocess.Popen(
                        [sys.executable, "-c", "import time; time.sleep(20)"],
                        **options,
                    )
                    children.append(child)
                    state["processes"].append(child)
                    yield "running"
                    raise RuntimeError("worker failed")

                namespace["run_preprocess_dataset"] = runner
                updates = namespace["preprocess_dataset"]("input", "experiment", "40k", 1)
                try:
                    next(updates)
                    self.assertEqual(next(updates)[0], "running")
                    if action == "close":
                        updates.close()
                    else:
                        self.assertIn("失败", list(updates)[-1][0])
                    self.assertIsNotNone(children[0].poll())
                    self.assertIsNone(namespace["TRAIN_TASK"])
                    self.assertEqual(namespace["begin_train_task"]("next task")[0], "start")
                finally:
                    updates.close()
                    for child in children:
                        if child.poll() is None:
                            child.kill()
                        child.wait(timeout=5)

    def test_cleanup_holds_slot_and_only_terminates_owned_processes(self):
        namespace = task_namespace()
        _, state = namespace["begin_train_task"]("owned task")
        child = mock.Mock()
        child.poll.return_value = None
        state["processes"].append(child)

        def terminate(process, *args):
            self.assertIs(process, child)
            self.assertIs(namespace["TRAIN_TASK"], state)
            self.assertEqual(namespace["begin_train_task"]("next task")[0], "busy")
            process.poll.return_value = -15

        namespace["kill_process_tree"].side_effect = terminate
        namespace["finish_train_task"](state)
        namespace["kill_process_tree"].assert_called_once_with(child, "owned task", namespace["logger"])
        self.assertIsNone(namespace["TRAIN_TASK"])
        namespace["finish_train_task"](state)
        self.assertEqual(namespace["kill_process_tree"].call_count, 1)

    def test_model_and_onekey_callbacks_clean_up_each_active_stage(self):
        model_args = ("experiment", "40k", True, 7, 1, 2, 1,
                      "是", "", "", "", "否", "否", "v2")
        onekey_args = ("experiment", "40k", True, "input", 7, 1, "rmvpe",
                       1, 2, 1, "是", "", "", "", "否", "否", "v2", "")
        stages = ("run_preprocess_dataset", "run_extract_f0_feature",
                  "run_train_model", "run_train_index")
        cases = [("click_train", "run_train_model", model_args)]
        cases += [("train1key", stage, onekey_args) for stage in stages]
        for callback, stage, args in cases:
            for action in ("close", "failure"):
                with self.subTest(callback=callback, stage=stage, action=action):
                    namespace = task_namespace()
                    namespace.update(
                        weight_names=lambda: [], change_choices=lambda: {},
                        refresh_weight_choices=lambda previous: (previous, {}),
                    )
                    for name in stages:
                        namespace[name] = lambda *args, **kwargs: iter(())
                    child = mock.Mock()
                    child.poll.return_value = None

                    def active_stage(*args, **kwargs):
                        namespace["TRAIN_TASK"]["processes"].append(child)
                        yield "active child"
                        raise RuntimeError("stage failed")

                    def terminate(*args):
                        child.poll.return_value = -15

                    namespace[stage] = active_stage
                    namespace["kill_process_tree"].side_effect = terminate
                    updates = namespace[callback](*args)
                    try:
                        for _ in range(8):
                            row = next(updates)
                            if "active child" in row[0]:
                                break
                        else:
                            self.fail("target stage did not start")
                        if action == "close":
                            updates.close()
                        else:
                            self.assertIn("失败", list(updates)[-1][0])
                        namespace["kill_process_tree"].assert_called_once()
                        self.assertIsNotNone(child.poll())
                        self.assertIsNone(namespace["TRAIN_TASK"])
                    finally:
                        updates.close()

    def test_failed_termination_keeps_stop_target_and_blocks_new_task(self):
        namespace = task_namespace()
        _, state = namespace["begin_train_task"]("owned task")
        child = mock.Mock()
        child.poll.return_value = None
        state["processes"].append(child)
        with self.assertRaisesRegex(RuntimeError, "did not exit"):
            namespace["finish_train_task"](state)
        self.assertIs(namespace["TRAIN_TASK"], state)
        self.assertEqual(namespace["begin_train_task"]("next task")[0], "busy")
        namespace["stop_train_task"]("owned task")
        self.assertEqual(namespace["kill_process_tree"].call_count, 2)


def dataset_fixture(root, with_f0, feature_size=256, sample_rate=16000):
    root = Path(root)
    prefix = "带空格 音频" + ("-f0" if with_f0 else "-plain")
    audio_path = root / (prefix + ".wav")
    samples = (0.3 * np.sin(2 * np.pi * 220 * np.arange(5000) / sample_rate)).astype(np.float32)
    wavfile.write(audio_path, sample_rate, samples)
    phone = np.arange(16 * feature_size, dtype=np.float32).reshape(16, feature_size) / 1000
    phone_path = root / (prefix + ".npy")
    np.save(phone_path, phone)
    record = [str(audio_path), str(phone_path)]
    if with_f0:
        pitch_path, pitchf_path = root / (prefix + "-pitch.npy"), root / (prefix + "-pitchf.npy")
        np.save(pitch_path, np.arange(32, dtype=np.int64) + 1)
        np.save(pitchf_path, np.arange(32, dtype=np.float32) + 100)
        record += [str(pitch_path), str(pitchf_path)]
    record += ["7"]
    filelist = root / (prefix + ".txt")
    filelist.write_text("|".join(record), encoding="utf-8")
    params = SimpleNamespace(
        max_wav_value=32768, sampling_rate=sample_rate,
        filter_length=512, hop_length=160, win_length=512,
    )
    dataset_class = data_utils.TextAudioLoaderMultiNSFsid if with_f0 else data_utils.TextAudioLoader
    return dataset_class(str(filelist), params), audio_path, samples, phone


class TrainingDatasetTests(unittest.TestCase):
    def test_dataset_fields_and_frame_alignment(self):
        for with_f0 in (False, True):
            for feature_size in (256, 768):
                with self.subTest(f0=with_f0, features=feature_size), tempfile.TemporaryDirectory() as root:
                    dataset, path, samples, phone = dataset_fixture(root, with_f0, feature_size)
                    self.assertEqual(len(dataset), 1)
                    self.assertEqual(dataset.lengths, [path.stat().st_size // (3 * 160)])
                    spec, wav, features, *labels = dataset[0]
                    frames = spec.shape[1]
                    self.assertEqual(features.shape, (frames, feature_size))
                    self.assertEqual(wav.shape, (1, frames * 160))
                    np.testing.assert_array_equal(wav.numpy()[0], samples[:frames * 160])
                    np.testing.assert_array_equal(features.numpy(), np.repeat(phone, 2, axis=0)[:frames])
                    self.assertEqual(labels[-1].item(), 7)
                    if with_f0:
                        torch.testing.assert_close(labels[0], torch.arange(frames) + 1)
                        torch.testing.assert_close(labels[1], torch.arange(frames).float() + 100)

    def test_valid_cache_is_reused_without_recomputing(self):
        for with_f0 in (False, True):
            with self.subTest(f0=with_f0), tempfile.TemporaryDirectory() as root:
                dataset, path, _, _ = dataset_fixture(root, with_f0)
                expected = dataset.get_audio(str(path))
                self.assertTrue(path.with_suffix(".spec.pt").is_file())
                with mock.patch.object(data_utils, "spectrogram_torch", side_effect=AssertionError("recomputed")):
                    actual = dataset.get_audio(str(path))
                for before, after in zip(expected, actual):
                    torch.testing.assert_close(after, before, rtol=0, atol=0)

    def test_corrupt_cache_is_regenerated(self):
        for with_f0 in (False, True):
            with self.subTest(f0=with_f0), tempfile.TemporaryDirectory() as root:
                dataset, path, _, _ = dataset_fixture(root, with_f0)
                expected, _ = dataset.get_audio(str(path))
                cache_path = path.with_suffix(".spec.pt")
                cache_path.write_bytes(b"invalid cache")
                with self.assertLogs("train.spec_cache", level="WARNING"):
                    actual, _ = dataset.get_audio(str(path))
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                torch.testing.assert_close(torch.load(cache_path, weights_only=True)["spec"],
                                           expected, rtol=0, atol=0)

    def test_sample_rate_mismatch_is_rejected_before_caching(self):
        for with_f0 in (False, True):
            with self.subTest(f0=with_f0), tempfile.TemporaryDirectory() as root:
                dataset, path, _, _ = dataset_fixture(root, with_f0)
                dataset.sampling_rate = 32000
                with self.assertRaisesRegex(ValueError, "16000 SR doesn't match target 32000 SR"):
                    dataset.get_audio(str(path))
                self.assertFalse(path.with_suffix(".spec.pt").exists())

    def test_replaced_audio_invalidates_cache_even_with_original_timestamp(self):
        for with_f0 in (False, True):
            with self.subTest(f0=with_f0), tempfile.TemporaryDirectory() as root:
                dataset, path, samples, _ = dataset_fixture(root, with_f0)
                original, _ = dataset.get_audio(str(path))
                stat = path.stat()
                wavfile.write(path, 16000, np.zeros_like(samples))
                os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
                actual, audio = dataset.get_audio(str(path))
                self.assertEqual(path.stat().st_size, stat.st_size)
                self.assertEqual(path.stat().st_mtime_ns, stat.st_mtime_ns)
                self.assertEqual(audio.abs().max().item(), 0)
                self.assertFalse(torch.equal(actual, original))
                self.assertLess(actual.max().item(), 0.001)

    def test_changed_frequency_parameters_invalidate_cache(self):
        for attribute, value in (("filter_length", 1024), ("hop_length", 80), ("win_length", 256)):
            for with_f0 in (False, True):
                with self.subTest(parameter=attribute, f0=with_f0), tempfile.TemporaryDirectory() as root:
                    dataset, path, _, _ = dataset_fixture(root, with_f0)
                    original, _ = dataset.get_audio(str(path))
                    setattr(dataset, attribute, value)
                    with mock.patch.object(data_utils, "spectrogram_torch", wraps=data_utils.spectrogram_torch) as compute:
                        actual, _ = dataset.get_audio(str(path))
                    compute.assert_called_once()
                    self.assertFalse(torch.equal(actual, original))

    def test_old_tensor_cache_is_recomputed_once(self):
        with tempfile.TemporaryDirectory() as root:
            dataset, path, _, _ = dataset_fixture(root, True)
            expected, _ = dataset.get_audio(str(path))
            cache = path.with_suffix(".spec.pt")
            torch.save(expected + 1, cache, _use_new_zipfile_serialization=False)
            with mock.patch.object(data_utils, "spectrogram_torch", wraps=data_utils.spectrogram_torch) as compute:
                actual, _ = dataset.get_audio(str(path))
                cached, _ = dataset.get_audio(str(path))
            compute.assert_called_once()
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            torch.testing.assert_close(cached, expected, rtol=0, atol=0)
            self.assertEqual(torch.load(cache, weights_only=True)["version"], spec_cache.CACHE_VERSION)

    def test_unverifiable_cache_payload_is_recomputed(self):
        for invalid in ("version", "metadata", "shape", "dtype", "nan"):
            with self.subTest(invalid=invalid), tempfile.TemporaryDirectory() as root:
                dataset, path, _, _ = dataset_fixture(root, False)
                expected, _ = dataset.get_audio(str(path))
                cache = path.with_suffix(".spec.pt")
                payload = torch.load(cache, weights_only=True)
                if invalid == "version":
                    payload["version"] += 1
                elif invalid == "metadata":
                    payload["metadata"]["sample_rate"] += 1
                elif invalid == "shape":
                    payload["spec"] = torch.ones(1, 1)
                elif invalid == "dtype":
                    payload["spec"] = expected.double()
                else:
                    payload["spec"].fill_(float("nan"))
                torch.save(payload, cache)
                actual, _ = dataset.get_audio(str(path))
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_failed_atomic_write_keeps_previous_cache_and_removes_temp_file(self):
        with tempfile.TemporaryDirectory() as root:
            dataset, path, samples, _ = dataset_fixture(root, False)
            dataset.get_audio(str(path))
            cache = path.with_suffix(".spec.pt")
            original = cache.read_bytes()
            wavfile.write(path, 16000, np.zeros_like(samples))
            with mock.patch.object(spec_cache.torch, "save", side_effect=OSError("disk full")):
                with self.assertRaisesRegex(OSError, "disk full"):
                    dataset.get_audio(str(path))
            self.assertEqual(cache.read_bytes(), original)
            self.assertFalse(any(p.name.endswith(".tmp") for p in Path(root).iterdir()))
            actual, _ = dataset.get_audio(str(path))
            self.assertLess(actual.max().item(), 0.001)

    def test_sharing_violation_retries_then_replaces_complete_cache(self):
        with tempfile.TemporaryDirectory() as root:
            dataset, path, _, _ = dataset_fixture(root, False)
            expected, _ = dataset.get_audio(str(path))
            cache = path.with_suffix(".spec.pt")
            signature = torch.load(cache, weights_only=True)["metadata"]
            replace = os.replace
            attempts = []

            def temporarily_locked(source, destination):
                attempts.append(destination)
                if len(attempts) < 3:
                    raise PermissionError("sharing violation")
                replace(source, destination)

            with mock.patch.object(spec_cache.os, "replace", side_effect=temporarily_locked), mock.patch.object(spec_cache.time, "sleep"):
                spec_cache.save_cached_spec(str(cache), expected, signature)
            self.assertEqual(len(attempts), 3)
            torch.testing.assert_close(torch.load(cache, weights_only=True)["spec"], expected, rtol=0, atol=0)
            self.assertFalse(any(p.name.endswith(".tmp") for p in Path(root).iterdir()))

    def test_concurrent_spawn_workers_build_one_complete_cache(self):
        with tempfile.TemporaryDirectory() as root:
            dataset, path, _, _ = dataset_fixture(root, True)
            collate = data_utils.TextAudioCollateMultiNSFsid()
            expected = collate([dataset[0]])
            path.with_suffix(".spec.pt").unlink()
            repeated = torch.utils.data.Subset(dataset, [0] * 8)
            batches = list(DataLoader(
                repeated, batch_size=1, collate_fn=collate, num_workers=2,
                multiprocessing_context="spawn", timeout=30,
            ))
            self.assertEqual(len(batches), 8)
            for batch in batches:
                for before, after in zip(expected, batch):
                    torch.testing.assert_close(after, before, rtol=0, atol=0)
            payload = torch.load(path.with_suffix(".spec.pt"), weights_only=True)
            self.assertEqual(payload["version"], spec_cache.CACHE_VERSION)
            self.assertFalse(any(p.name.endswith(".tmp") for p in Path(root).iterdir()))

    def test_spawn_worker_matches_inprocess_batch(self):
        for with_f0 in (False, True):
            with self.subTest(f0=with_f0), tempfile.TemporaryDirectory() as root:
                dataset, _, _, _ = dataset_fixture(root, with_f0)
                collate = data_utils.TextAudioCollateMultiNSFsid() if with_f0 else data_utils.TextAudioCollate()
                expected = list(DataLoader(dataset, batch_size=1, collate_fn=collate, num_workers=0))
                actual = list(DataLoader(
                    dataset, batch_size=1, collate_fn=collate, num_workers=1,
                    multiprocessing_context="spawn", timeout=30,
                ))
                self.assertEqual(len(actual), len(expected))
                for before, after in zip(expected[0], actual[0]):
                    torch.testing.assert_close(after, before, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
