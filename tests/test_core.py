"""Unit tests for the parts where a silent bug would poison every embedding.

    python -m pytest tests -q      (or: python tests/test_core.py)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from facelib.align.umeyama import apply_affine, invert_affine, similarity_matrix_2x3, umeyama
from facelib.align.warp import alignment_error, reference_landmarks
from facelib.detect.nms import nms, soft_nms
from facelib.pipeline.gallery import Gallery


def test_umeyama_recovers_known_transform():
    rng = np.random.default_rng(0)
    src = rng.uniform(0, 100, (5, 2))
    angle, scale, t = np.deg2rad(23.0), 1.7, np.array([12.0, -8.0])
    R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    dst = (scale * src @ R.T) + t
    M = similarity_matrix_2x3(src, dst)
    assert np.allclose(apply_affine(src, M), dst, atol=1e-4)
    # recovered scale = sqrt(det) of the linear part
    assert abs(np.sqrt(abs(np.linalg.det(M[:, :2]))) - scale) < 1e-4


def test_umeyama_no_reflection():
    """A mirrored point set must not be fitted with a reflection."""
    src = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    dst = np.array([[0.0, 0.0], [-1.0, 0.0], [0.0, 1.0]])
    M = umeyama(src, dst)[:2]
    assert np.linalg.det(M[:, :2]) > 0


def test_affine_inverse_roundtrip():
    pts = np.array([[10.0, 20.0], [30.0, 40.0]])
    M = similarity_matrix_2x3(pts, pts * 2.0 + 5.0)
    back = apply_affine(apply_affine(pts, M), invert_affine(M))
    assert np.allclose(back, pts, atol=1e-4)


def test_alignment_error_zero_on_template():
    ref = reference_landmarks(112)
    assert alignment_error(ref, 112) < 1e-3


def test_anchor_encode_decode_roundtrip():
    import torch

    from facelib.detect.anchors import (center_to_corner, decode_boxes, decode_landmarks,
                                        encode_boxes, encode_landmarks, generate_anchors)
    anchors = generate_anchors((128, 128))
    boxes = center_to_corner(anchors[:50]) * 0.97 + 0.005   # perturb so it is a real fit
    enc = encode_boxes(boxes, anchors[:50])
    dec = decode_boxes(enc, anchors[:50])
    assert torch.allclose(dec, boxes, atol=1e-5)

    lmk = torch.rand(50, 10) * 0.5 + 0.25
    assert torch.allclose(decode_landmarks(encode_landmarks(lmk, anchors[:50]), anchors[:50]),
                          lmk, atol=1e-5)


def test_anchor_count_matches_head_output():
    """Anchor ordering/count must equal what the detector heads emit, or every box is wrong."""
    import torch

    from facelib.detect.anchors import generate_anchors, num_anchors_for
    from facelib.detect.model import build_detector
    model = build_detector({"width": 0.25, "fpn_channels": 64}).eval()
    with torch.no_grad():
        loc, logits, lmk = model(torch.zeros(1, 3, 128, 128))
    expected = num_anchors_for((128, 128))
    assert loc.shape == (1, expected, 4)
    assert logits.shape == (1, expected, 2)
    assert lmk.shape == (1, expected, 10)
    assert generate_anchors((128, 128)).shape[0] == expected


def test_nms_removes_duplicates_keeps_distinct():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [100, 100, 110, 110]], np.float32)
    scores = np.array([0.9, 0.8, 0.7], np.float32)
    keep = nms(boxes, scores, 0.4)
    assert keep.tolist() == [0, 2]
    k, s = soft_nms(boxes, scores)
    assert 0 in k.tolist() and 2 in k.tolist() and len(s) == len(k)


def test_margin_head_penalises_target_logit():
    import torch

    from facelib.nn.margins import CombinedMarginHead
    head = CombinedMarginHead(8, 4, scale=1.0, m2=0.5).eval()
    emb = torch.randn(6, 8)
    labels = torch.tensor([0, 1, 2, 3, 0, 1])
    cos = head.cosine(emb)
    logits = head(emb, labels)
    idx = torch.arange(6)
    # The margin must lower the ground-truth logit and leave the others untouched.
    assert (logits[idx, labels] < cos[idx, labels] + 1e-6).all()
    mask = torch.ones_like(cos, dtype=torch.bool)
    mask[idx, labels] = False
    assert torch.allclose(logits[mask], cos[mask], atol=1e-5)


def test_margin_warmup_ramps():
    import torch

    from facelib.nn.margins import CombinedMarginHead
    head = CombinedMarginHead(8, 3, scale=1.0, m2=0.5, margin_warmup_steps=100).train()
    emb, labels = torch.randn(4, 8), torch.tensor([0, 1, 2, 0])
    first = head(emb, labels)[torch.arange(4), labels].clone()
    for _ in range(200):
        head(emb, labels)
    later = head(emb, labels)[torch.arange(4), labels]
    assert (later < first).all()  # margin grew, so the target logit dropped further


def test_embedding_backbones_output_shape():
    import torch

    from facelib.nn.builder import NormalizedEmbedding, build_embedding_model
    for name in ("mobilefacenet", "iresnet18"):
        m = build_embedding_model({"name": name, "embedding_dim": 64, "input_size": 112,
                                   "width": 0.5 if name == "mobilefacenet" else 0.25}).eval()
        with torch.no_grad():
            out = NormalizedEmbedding(m)(torch.randn(2, 3, 112, 112))
        assert out.shape == (2, 64)
        assert torch.allclose(out.norm(dim=1), torch.ones(2), atol=1e-4)


def test_verification_metrics_perfect_separation():
    from facelib.engine.evaluate import verification_metrics
    scores = np.concatenate([np.full(50, 0.9), np.full(50, 0.1)])
    labels = np.concatenate([np.ones(50, int), np.zeros(50, int)])
    m = verification_metrics(scores, labels, folds=5)
    assert m["accuracy"] > 0.99 and m["auc"] > 0.99 and m["eer"] < 0.01


def test_gallery_identify_and_roundtrip(tmp_path=None):
    tmp = Path(tmp_path) if tmp_path else Path(".")
    g = Gallery(4, mode="centroid", threshold=0.5)
    g.add("a", np.array([1, 0, 0, 0], np.float32))
    g.add("a", np.array([0.9, 0.1, 0, 0], np.float32))
    g.add("b", np.array([0, 1, 0, 0], np.float32))
    assert g.identify(np.array([1, 0.05, 0, 0], np.float32)).label == "a"
    assert g.identify(np.array([0, 0, 1, 0], np.float32)).label == "unknown"
    p = tmp / "g.npz"
    g.save(p)
    g2 = Gallery.load(p)
    assert g2.summary()["identities"] == 2
    assert g2.identify(np.array([0, 1, 0, 0], np.float32)).label == "b"
    p.unlink()


def test_transforms_normalization_range():
    from facelib.data.transforms import build_train_transform, preprocess_batch
    img = np.random.randint(0, 256, (112, 112, 3), dtype=np.uint8)
    t = build_train_transform((112, 112), "heavy")(img)
    assert t.shape == (3, 112, 112) and -1.1 <= float(t.min()) and float(t.max()) <= 1.1
    batch = preprocess_batch([img, img])
    assert tuple(batch.shape) == (2, 3, 112, 112)


if __name__ == "__main__":
    import inspect

    mod = sys.modules[__name__]
    tests = [(n, f) for n, f in vars(mod).items() if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            if "tmp_path" in inspect.signature(fn).parameters:
                fn(Path("."))
            else:
                fn()
            print("PASS %s" % name)
        except Exception as exc:
            failed += 1
            print("FAIL %s: %s: %s" % (name, type(exc).__name__, exc))
    print("\n%d/%d passed" % (len(tests) - failed, len(tests)))
    raise SystemExit(1 if failed else 0)
