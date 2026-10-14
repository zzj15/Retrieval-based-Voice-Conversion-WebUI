import tempfile
import unittest
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import av
import numpy as np
import soundfile as sf
import torch

from infer.audio import write_audio
from infer.realtime_audio import mix_rms, sola_crossfade


class AudioWritingTests(unittest.TestCase):
    def test_lossless_formats_preserve_pcm_samples_and_channels(self):
        with tempfile.TemporaryDirectory() as root:
            samples = np.arange(-1024, 1024, dtype=np.int16)
            for channels in (1, 2):
                audio = samples if channels == 1 else np.column_stack((samples, -samples))
                for extension in ("wav", "flac"):
                    with self.subTest(channels=channels, format=extension):
                        path = Path(root, "带空格 输出", f"音频 {channels}.{extension}")
                        write_audio(str(path), audio, 16000, extension)
                        decoded, rate = sf.read(path, dtype="int16")
                        self.assertEqual(rate, 16000)
                        self.assertEqual(sf.info(path).channels, channels)
                        self.assertEqual(sf.info(path).subtype, "PCM_16")
                        np.testing.assert_array_equal(decoded, audio)

    def test_compressed_formats_keep_rate_channels_and_complete_audio(self):
        phase = np.arange(16000) / 16000
        audio = np.column_stack((12000 * np.sin(2 * np.pi * 220 * phase),
                                 8000 * np.sin(2 * np.pi * 440 * phase))).astype(np.int16)
        with tempfile.TemporaryDirectory() as root:
            for extension in ("mp3", "m4a"):
                with self.subTest(format=extension):
                    path = Path(root, f"compressed.{extension}")
                    write_audio(path, audio, 16000, extension)
                    with av.open(str(path)) as container:
                        stream = container.streams.audio[0]
                        self.assertEqual(stream.codec_context.sample_rate, 16000)
                        self.assertEqual(stream.codec_context.channels, 2)
                        frames = list(container.decode(stream))
                    self.assertTrue(frames)
                    sample_count = sum(frame.samples for frame in frames)
                    self.assertGreaterEqual(sample_count, 16000 - 1024)
                    self.assertLessEqual(sample_count, 16000 + 2048)
                    decoded = np.concatenate([frame.to_ndarray().ravel() for frame in frames])
                    self.assertTrue(np.isfinite(decoded).all())
                    self.assertGreater(np.max(np.abs(decoded)), 0.01)

    def test_encoder_failure_propagates_and_closes_output(self):
        import builtins

        handles = []
        original_open = builtins.open

        def record_open(*args, **kwargs):
            handle = original_open(*args, **kwargs)
            handles.append(handle)
            return handle

        with tempfile.TemporaryDirectory() as root, mock.patch(
            "infer.audio.wav2", side_effect=RuntimeError("encoder failed")
        ), mock.patch("builtins.open", side_effect=record_open):
            with self.assertRaisesRegex(RuntimeError, "encoder failed"):
                write_audio(Path(root, "error.m4a"), np.zeros(1024), 16000, "m4a")
        self.assertTrue(handles)
        self.assertTrue(all(handle.closed for handle in handles))


class RealtimeAudioTests(unittest.TestCase):
    def test_rms_mixes_volume_with_expected_gain_in_place(self):
        phase = torch.arange(1600) / 16000
        source = 0.2 * torch.sin(2 * torch.pi * 220 * phase)
        original = source * 0.5
        for mix_rate in (0.0, 0.5, 1.0):
            with self.subTest(mix=mix_rate):
                output = original.clone()
                result = mix_rms(source, output, 160, mix_rate)
                self.assertIs(result, output)
                torch.testing.assert_close(output, original * (2 ** (1 - mix_rate)))

    def test_silent_source_or_output_remains_finite(self):
        silence = torch.zeros(1600)
        tone = torch.full((1600,), 0.1)
        for source, output in ((silence, tone.clone()), (tone, silence.clone()),
                               (silence, silence.clone())):
            result = mix_rms(source, output, 160, 0.5)
            self.assertTrue(torch.isfinite(result).all())
            torch.testing.assert_close(result, silence, rtol=0, atol=0)

    def test_sola_aligns_known_overlap_and_advances_buffer(self):
        generator = torch.Generator().manual_seed(1234)
        buffer = torch.randn(16, generator=generator)
        original_buffer = buffer.clone()
        audio = torch.randn(96, generator=generator) * 0.01
        audio[8:24] = buffer
        original_audio = audio.clone()
        fade_in = torch.sin(0.5 * torch.pi * torch.linspace(0, 1, 16)) ** 2
        aligned, offset = sola_crossfade(
            audio, buffer, torch.ones(1, 1, 16), fade_in, 1 - fade_in, 64, 16
        )
        self.assertEqual(offset, 8)
        torch.testing.assert_close(aligned[:16], original_buffer)
        torch.testing.assert_close(aligned[16:], original_audio[24:], rtol=0, atol=0)
        torch.testing.assert_close(buffer, original_audio[72:88], rtol=0, atol=0)
        self.assertEqual(aligned.data_ptr(), audio.data_ptr() + 8 * audio.element_size())

    def test_sola_starts_at_zero_with_empty_history(self):
        audio = torch.linspace(-0.1, 0.1, 96)
        original = audio.clone()
        buffer = torch.zeros(16)
        fade_in = torch.linspace(0, 1, 16)
        aligned, offset = sola_crossfade(
            audio, buffer, torch.ones(1, 1, 16), fade_in, 1 - fade_in, 64, 16
        )
        self.assertEqual(offset, 0)
        torch.testing.assert_close(aligned[:16], original[:16] * fade_in, rtol=0, atol=0)
        torch.testing.assert_close(buffer, original[64:80], rtol=0, atol=0)

    def test_sola_keeps_blended_history_when_overlap_exceeds_block(self):
        audio = torch.full((96,), 0.1)
        buffer = torch.full((64,), 0.2)
        fade_in = torch.linspace(0, 1, 64)
        aligned, offset = sola_crossfade(
            audio, buffer, torch.ones(1, 1, 64), fade_in, 1 - fade_in, 16, 16
        )
        self.assertEqual(offset, 0)
        expected_fade = torch.linspace(0.2, 0.1, 64)
        torch.testing.assert_close(aligned[:64], expected_fade)
        expected_history = torch.cat((expected_fade[16:], torch.full((16,), 0.1)))
        torch.testing.assert_close(buffer, expected_history)

    def test_vst_noise_gate_accepts_context_samples_and_keeps_block_length(self):
        root = Path(__file__).resolve().parents[1]
        spec = importlib.util.spec_from_file_location(
            "worker_audio_test", root / "RVCRealtimeVST/worker/rvc_worker.py"
        )
        worker = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(worker)

        def model_stub(*args):
            return SimpleNamespace(
                tgt_sr=16000, change_key=lambda value: None,
                change_formant=lambda value: None, change_index_rate=lambda value: None,
                infer=lambda source, block, skip, length, method: torch.zeros(length * 160),
            )

        with mock.patch("configs.config.Config", return_value=SimpleNamespace(device="cpu")), mock.patch(
            "infer.rtrvc.RVC", side_effect=model_stub
        ):
            for rate in (16000, 44100, 48000):
                with self.subTest(rate=rate):
                    engine = worker.RVCStreamEngine(dict(
                        rvc_root=str(root), sample_rate=rate, block_ms=100,
                        crossfade_ms=30, extra_ms=100, model_path="fixture.pth", index_path="",
                    ))
                    for amplitude in (0.1, 0.0, 0.1):
                        result = engine.process(
                            np.full(engine.block_frame, amplitude, np.float32),
                            0, 0, 0, 0.5, -30, 0,
                        )
                        self.assertEqual(result.shape, (engine.block_frame,))
                        self.assertTrue(np.isfinite(result).all())


if __name__ == "__main__":
    unittest.main()
