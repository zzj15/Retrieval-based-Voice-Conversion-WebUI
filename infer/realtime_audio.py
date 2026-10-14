"""Shared waveform processing for the realtime GUI and VST worker."""

import librosa
import torch
from torch.nn import functional as F


def _rms_envelope(audio, length, zc, device):
    rms = librosa.feature.rms(
        y=audio[:length].cpu().numpy(), frame_length=4 * zc, hop_length=zc
    )
    rms = torch.from_numpy(rms).to(device)
    return F.interpolate(
        rms.unsqueeze(0), size=length + 1, mode="linear", align_corners=True
    )[0, 0, :-1]


def mix_rms(input_audio, output_audio, zc, mix_rate):
    """Mix the input volume envelope into the output waveform in place."""
    length = output_audio.shape[0]
    rms1 = _rms_envelope(input_audio, length, zc, output_audio.device)
    rms2 = _rms_envelope(output_audio, length, zc, output_audio.device)
    rms2 = torch.maximum(rms2, torch.full_like(rms2, 1e-3))
    output_audio *= torch.pow(rms1 / rms2, 1.0 - mix_rate)
    return output_audio


# SOLA algorithm from https://github.com/yxlllc/DDSP-SVC
def sola_crossfade(
    audio, buffer, denominator_kernel, fade_in, fade_out, block_frame, search_frame
):
    """Align and blend a block, updating the overlap buffer in place."""
    buffer_frame = buffer.shape[0]
    conv_input = audio[None, None, : buffer_frame + search_frame]
    numerator = F.conv1d(conv_input, buffer[None, None, :])
    denominator = torch.sqrt(F.conv1d(conv_input**2, denominator_kernel) + 1e-8)
    offset = int(torch.argmax(numerator[0, 0] / denominator[0, 0]))
    audio = audio[offset:]
    audio[:buffer_frame] *= fade_in
    audio[:buffer_frame] += buffer * fade_out
    buffer[:] = audio[block_frame : block_frame + buffer_frame]
    return audio, offset
