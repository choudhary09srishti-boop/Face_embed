"""Sweep the search threshold on the demo set: hit-rate, precision, false-alarm rate."""
import csv

import numpy as np

emb = np.load("out/demo_embeddings.npy")
emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
rows = list(csv.DictReader(open("out/demo_embeddings.csv", newline="", encoding="utf-8")))
truth = np.array([r["image"].replace("\\", "/").split("/")[-1].split("__")[0] for r in rows])

sims = emb @ emb.T
n = len(truth)

print("photos: %d, people: %d" % (n, len(set(truth))))
print("%5s %9s %10s %12s" % ("thr", "hit-rate", "precision", "false-alarm"))
for t in np.arange(0.20, 0.85, 0.05):
    present_q = hits = tp = returned = false_alarm = 0
    for i in range(n):
        not_self = np.arange(n) != i
        same = (truth == truth[i]) & not_self
        # Case 1: person IS in the database (their other photos are there)
        if same.any():
            present_q += 1
            got = not_self & (sims[i] >= t)
            hits += int((got & same).any())
            tp += int((got & same).sum())
            returned += int(got.sum())
        # Case 2: person is NOT in the database (all their photos removed)
        absent_db = truth != truth[i]
        false_alarm += int((sims[i][absent_db] >= t).any())
    print("%5.2f %9.3f %10.3f %12.3f" % (
        t, hits / max(present_q, 1), tp / max(returned, 1), false_alarm / n))