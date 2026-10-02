"""Sweep clustering settings on the demo set and score them against filename ground truth."""
import csv

import numpy as np
from sklearn.cluster import DBSCAN, AgglomerativeClustering

emb = np.load("out/demo_embeddings.npy")
emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
rows = list(csv.DictReader(open("out/demo_embeddings.csv", newline="", encoding="utf-8")))
truth = np.array([r["image"].replace("\\", "/").split("/")[-1].split("__")[0] for r in rows])

same_true = truth[:, None] == truth[None, :]
iu = np.triu_indices(len(truth), k=1)


def score(labels):
    labels = labels.copy()
    noise = labels == -1
    labels[noise] = -(np.arange(noise.sum()) + 2)  # each unclustered face = its own group
    same_pred = labels[:, None] == labels[None, :]
    tp = (same_pred & same_true)[iu].sum()
    fp = (same_pred & ~same_true)[iu].sum()
    fn = (~same_pred & same_true)[iu].sum()
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    return p, r, 2 * p * r / max(p + r, 1e-9)


print("true people: %d, photos: %d" % (len(set(truth)), len(truth)))
print("%-8s %5s %9s %6s %6s %6s" % ("method", "thr", "clusters", "prec", "rec", "f1"))
for t in np.arange(0.40, 0.95, 0.05):
    for name in ("dbscan", "average"):
        if name == "dbscan":
            lab = DBSCAN(eps=t, min_samples=2, metric="cosine").fit_predict(emb)
        else:
            lab = AgglomerativeClustering(n_clusters=None, distance_threshold=t,
                                          metric="cosine", linkage="average").fit_predict(emb)
        n = len(set(lab)) - (1 if -1 in lab else 0)
        p, r, f1 = score(lab)
        print("%-8s %5.2f %9d %6.3f %6.3f %6.3f" % (name, t, n, p, r, f1))