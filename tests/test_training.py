import ast
import logging
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
    }
    tree = ast.parse(source_path.read_text())
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
                process = object()

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
                with self.assertLogs("train.data_utils", level="WARNING"):
                    actual, _ = dataset.get_audio(str(path))
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                torch.testing.assert_close(torch.load(cache_path), expected, rtol=0, atol=0)

    def test_sample_rate_mismatch_is_rejected_before_caching(self):
        for with_f0 in (False, True):
            with self.subTest(f0=with_f0), tempfile.TemporaryDirectory() as root:
                dataset, path, _, _ = dataset_fixture(root, with_f0)
                dataset.sampling_rate = 32000
                with self.assertRaisesRegex(ValueError, "16000 SR doesn't match target 32000 SR"):
                    dataset.get_audio(str(path))
                self.assertFalse(path.with_suffix(".spec.pt").exists())

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
