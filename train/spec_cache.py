"""Versioned spectrogram caches derived from the waveform actually read."""

import hashlib
import logging
import os
import tempfile
import time

import torch


logger = logging.getLogger(__name__)
CACHE_VERSION = 1


def audio_signature(audio, sample_rate, n_fft, hop_size, win_size):
    return {
        "audio_sha256": hashlib.sha256(memoryview(audio.contiguous().numpy())).hexdigest(),
        "audio_shape": list(audio.shape),
        "audio_dtype": str(audio.dtype),
        "sample_rate": int(sample_rate),
        "n_fft": int(n_fft),
        "hop_size": int(hop_size),
        "win_size": int(win_size),
        "center": False,
    }


def load_cached_spec(path, signature):
    try:
        cache = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(cache, dict) or cache.get("version") != CACHE_VERSION:
            return None
        if cache.get("metadata") != signature:
            return None
        spec = cache.get("spec")
        shape = signature["audio_shape"]
        if len(shape) != 1:
            return None
        n_fft, hop_size = signature["n_fft"], signature["hop_size"]
        padding = int((n_fft - hop_size) / 2)
        frames = (shape[0] + 2 * padding - n_fft) // hop_size + 1
        if (
            not isinstance(spec, torch.Tensor)
            or spec.dtype != torch.float32
            or tuple(spec.shape) != (n_fft // 2 + 1, frames)
            or not torch.isfinite(spec).all()
        ):
            return None
        return spec
    except FileNotFoundError:
        return None
    except Exception:
        logger.warning("Unable to read spectrogram cache %s; rebuilding it", path, exc_info=True)
        return None


def save_cached_spec(path, spec, signature):
    directory = os.path.dirname(path) or "."
    fd, temporary = tempfile.mkstemp(prefix="." + os.path.basename(path) + ".",
                                     suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "wb") as output:
            torch.save({"version": CACHE_VERSION, "metadata": signature, "spec": spec},
                       output, _use_new_zipfile_serialization=False)
            output.flush()
            os.fsync(output.fileno())
        # Windows readers can briefly prevent replacement of an open file.
        for attempt in range(5):
            try:
                os.replace(temporary, path)
                break
            except PermissionError:
                if attempt == 4:
                    raise
                time.sleep(0.02)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
