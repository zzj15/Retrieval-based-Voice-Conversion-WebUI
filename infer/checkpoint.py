"""RVC checkpoint loading and speaker metadata shared by the entry points."""


def normalize_speaker_info(speaker_info, speaker_count=110):
    speakers = []
    seen = set()
    for item in speaker_info or []:
        try:
            speaker_id = int(item["id"])
            speaker_name = str(item["name"])
        except (KeyError, TypeError, ValueError):
            continue
        if not 0 <= speaker_id < speaker_count or not speaker_name or speaker_id in seen:
            continue
        seen.add(speaker_id)
        speakers.append({"id": speaker_id, "name": speaker_name})
    return sorted(speakers, key=lambda item: item["id"])


def get_speaker_count(checkpoint):
    weight = checkpoint.get("weight", {}) if isinstance(checkpoint, dict) else {}
    embedding = weight.get("emb_g.weight")
    if embedding is None:
        raise ValueError("Model does not contain weight/emb_g.weight")
    return int(embedding.shape[0])


def load_checkpoint(path):
    import torch

    return torch.load(str(path), map_location="cpu")


def load_synthesizer(path, device="cpu", *, is_half=False, remove_weight_norm=False):
    checkpoint = load_checkpoint(path)
    model = build_synthesizer(
        checkpoint, device, is_half=is_half, remove_weight_norm=remove_weight_norm
    )
    return model, checkpoint


def build_synthesizer(checkpoint, device="cpu", *, is_half=False, remove_weight_norm=False):
    from infer.module.models import (
        SynthesizerTrnMs256NSFsid,
        SynthesizerTrnMs256NSFsid_nono,
        SynthesizerTrnMs768NSFsid,
        SynthesizerTrnMs768NSFsid_nono,
    )

    checkpoint["config"][-3] = get_speaker_count(checkpoint)
    model_type = (checkpoint.get("version", "v1"), checkpoint.get("f0", 1))
    synthesizers = {
        ("v1", 1): SynthesizerTrnMs256NSFsid,
        ("v1", 0): SynthesizerTrnMs256NSFsid_nono,
        ("v2", 1): SynthesizerTrnMs768NSFsid,
        ("v2", 0): SynthesizerTrnMs768NSFsid_nono,
    }
    try:
        synthesizer = synthesizers[model_type]
    except KeyError:
        raise ValueError("Unsupported RVC model version/F0: %r" % (model_type,)) from None
    model = synthesizer(*checkpoint["config"], is_half=is_half)
    del model.enc_q
    model.load_state_dict(checkpoint["weight"], strict=False)
    model.eval().to(device)
    model = model.half() if is_half else model.float()
    if remove_weight_norm:
        model.remove_weight_norm()
    return model
