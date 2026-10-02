"""Build a ~100-photo demo set with repeated people from held-out identities."""
import json
import random
import shutil
from pathlib import Path

RAW = Path("data/raw")
OUT = Path("data/demo/photos")
TARGET = 100

hold = json.loads(Path("runs/mbf2/holdout_identities.json").read_text(encoding="utf-8"))
random.Random(0).shuffle(hold)
OUT.mkdir(parents=True, exist_ok=True)

count, people = 0, 0
for name in hold:
    files = sorted((RAW / name).glob("*.jpg"))
    if len(files) < 3:
        continue
    people += 1
    for i, f in enumerate(files[:4]):
        if count >= TARGET:
            break
        shutil.copy(f, OUT / ("%s__%d.jpg" % (name, i)))
        count += 1
    if count >= TARGET:
        break

print("%d photos of %d people -> %s" % (count, people, OUT))