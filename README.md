# face_embed — face embedding model, trained from scratch

Give it an image, get back a **512-d L2-normalised embedding per face** (one image can
contain many faces). Same-person embeddings have high cosine similarity; different
people have low. That is what makes recognition, clustering and dedup possible
without retraining when a new person shows up.

Everything is implemented in this repo: backbones, the ArcFace angular-margin head,
the face detector with landmark regression, the alignment math, the training loop
and the evaluation protocol. No pretrained weights, no `insightface` / `facenet-pytorch` /
`dlib`. PyTorch, NumPy and OpenCV are used as the tensor/imaging substrate only.
(The single exception is OpenCV's bundled Haar cascade, used *optionally* to bootstrap
your first aligned dataset before your own detector is trained — see step 2.)

## Pipeline

```
image ──▶ detector (TinyFace: boxes + 5 landmarks)
            └─▶ similarity-align each face to a 112×112 canonical template
                  └─▶ backbone (MobileFaceNet or IResNet)
                        └─▶ 512-d embedding, L2-normalised
                              └─▶ cosine similarity ──▶ verify / identify / cluster
```

## Layout

| Path | What |
| --- | --- |
| `facelib/nn/` | `iresnet` (18–200), `mobilefacenet`, `margins` (ArcFace/CosFace/SphereFace + sub-centers), losses, layers |
| `facelib/detect/` | anchors + encode/decode/matching, NMS & soft-NMS, TinyFace model, multi-task loss, WIDER FACE reader, inference, bootstrap detectors |
| `facelib/align/` | Umeyama similarity transform, canonical 112×112 alignment |
| `facelib/data/` | augmentations, dataset with cached index, P×K sampler, verification pairs |
| `facelib/engine/` | embedding trainer, detector trainer, verification evaluation |
| `facelib/pipeline/` | `FaceEmbedder` (image → faces), `Gallery` (enroll → identify) |
| `scripts/` | every CLI entry point |

## Cloud setup

```bash
git clone <this> face_embed && cd face_embed
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
python scripts/smoke_test.py --out runs/smoke      # ~2 min, synthetic data, proves the wiring
python tests/test_core.py                          # unit tests
```

## 1. Data

Your photos, one directory per person:

```
data/raw/
  alice/ img1.jpg img2.jpg ...
  bob/   ...
```

**How much you need.** This is the part that decides whether the project works:

| Identities × images | Realistic outcome |
| --- | --- |
| < 100 ids | Train nothing useful from scratch. The embedding will not generalise past your own photos. |
| 1k–10k ids, 20+ imgs each | `mobilefacenet`, usable in-domain embeddings (~95–98% on your own holdout). |
| 10k–100k ids | `iresnet50`, genuinely good embeddings. |
| MS1M / Glint360k scale (100k+ ids, 5M+ imgs) | `iresnet100`, state-of-the-art territory. |

Identity **count** matters far more than images per identity — the margin loss learns
by separating identities. 10k people × 10 photos beats 100 people × 1000 photos.

## 2. Align the crops

The backbone only ever sees aligned 112×112 crops, so build them once up front:

```bash
# already tight crops:
python scripts/prepare_dataset.py --raw data/raw --out data/aligned --detector whole
# raw photos, bootstrap with opencv's Haar cascade + eye alignment:
python scripts/prepare_dataset.py --raw data/raw --out data/aligned --detector haar
# best: your own trained detector (step 5)
python scripts/prepare_dataset.py --raw data/raw --out data/aligned \
    --detector runs/detector/last.pt --device cuda --max-align-error 3.0
```

## 3. Train the embedding model

```bash
python scripts/train_embedding.py --config configs/embedding_mbf.yaml data.root=data/aligned
# bigger backbone / multi-GPU:
torchrun --nproc_per_node=4 scripts/train_embedding.py --config configs/embedding_r50.yaml
```

Any config key can be overridden on the command line: `optim.lr=0.05 train.epochs=40
head.name=cosface head.scale=32`.

Each epoch the trainer holds out **whole identities**, builds verification pairs from
them, and reports 10-fold accuracy, AUC, EER and TAR@FAR. Watch that number, not the
classification accuracy — training accuracy near 100% with flat verification accuracy
means memorisation. Checkpoints: `runs/<name>/last.pt` and `best.pt` (best verification).

Tuning notes that matter:
- **`head.scale`** 64 is for 100k+ identities; use 32 (or 16) for small sets.
- **`head.name`** `arcface` is the default; `cosface` is more forgiving of noisy labels;
  `head.sub_centers=3` also helps with label noise.
- **`data.augment`** `heavy` if your inference images are lower quality than training.
- Loss plateaus at the start → raise `head.margin_warmup_steps`.

## 4. Evaluate and calibrate the threshold

```bash
python scripts/make_pairs.py --root data/aligned --out data/pairs.txt --holdout-frac 0.1
python scripts/evaluate.py --checkpoint runs/mbf/best.pt --pairs data/pairs.txt --root data/aligned
```

Use `tar@far0.001_threshold` from the output as your production cosine threshold: it is
the cutoff at which only 1 impostor in 1,000 is accepted. Do not reuse a threshold from
a different model or dataset.

## 5. Train the detector (optional but recommended)

```bash
# WIDER FACE + the retinaface-style label.txt with 5 landmarks
python scripts/train_detector.py --config configs/detector.yaml
```

Without it, use `--detector haar` (fewer, frontal-only faces, coarser alignment) or
`--detector whole` for pre-cropped input.

## 6. Get embeddings

```bash
# batch: <out>.npy (N×512) + <out>.csv (one row per face)
python scripts/embed_images.py --checkpoint runs/mbf/best.pt --images photos/ \
    --detector runs/detector/last.pt --out out/embeddings --flip-tta
```

```python
from facelib.pipeline.embedder import FaceEmbedder

emb = FaceEmbedder("runs/mbf/best.pt", detector="runs/detector/last.pt", device="cuda")
for face in emb.embed_path("group_photo.jpg"):
    print(face.bbox, face.det_score, face.embedding.shape)   # (512,), unit norm
```

## 7. Recognise people

```bash
python scripts/gallery_cli.py enroll --checkpoint runs/mbf/best.pt \
    --images data/enroll --out weights/gallery.npz --threshold 0.35
python scripts/gallery_cli.py identify --checkpoint runs/mbf/best.pt \
    --gallery weights/gallery.npz --image group.jpg --draw out/annotated.jpg
```

Adding a person never requires retraining — enrolment is one forward pass.

## 8. Serve / export

```bash
EMBED_CHECKPOINT=runs/mbf/best.pt DETECTOR=runs/detector/last.pt GALLERY=weights/gallery.npz \
  uvicorn scripts.serve:app --host 0.0.0.0 --port 8000
curl -F file=@group.jpg localhost:8000/embed       # /embed /identify /compare /health
```

```bash
python scripts/export_onnx.py --checkpoint runs/mbf/best.pt --out weights/embedding.onnx
```

ONNX contract: input `NCHW` float32, RGB, `(pixel - 127.5) / 128.0`, 112×112;
output `embedding`, already L2-normalised.

## Things that will silently ruin results

1. **Different alignment at train vs inference.** Same template, same crop size, always.
2. **Reporting the threshold you tuned on the same pairs.** The evaluator cross-validates;
   keep it that way.
3. **Splitting train/val by image instead of by identity.** Then val accuracy is meaningless.
4. **Comparing un-normalised embeddings.** The pipeline normalises; if you touch raw
   backbone output, normalise it yourself.
5. **Too few identities.** No amount of training fixes it — see the table in step 1.

6. image Dataset https://www.kaggle.com/datasets/kaustubhdhote/human-faces-dataset
