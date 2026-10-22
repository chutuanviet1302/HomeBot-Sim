"""Dump every GET response of the API to web/frontend/public/data/*.json (and the media to public/media/) for a
static build of the site (npm run build:static).

python -m web.backend.export_static                # without the 31 MB demo video (keeps the build small)
python -m web.backend.export_static --with-video
"""
from __future__ import annotations

import json
import shutil
import sys

from . import app as api
from .db import ROOT

PUBLIC = ROOT / "web" / "frontend" / "public"
DATA = PUBLIC / "data"
MEDIA = PUBLIC / "media"


def dump(name: str, obj) -> None:
    (DATA / f"{name}.json").write_text(json.dumps(obj, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")


def main():
    if DATA.exists():
        shutil.rmtree(DATA)
    DATA.mkdir(parents=True)
    dump("kpi", api.kpi())
    dump("experiments", api.experiments())
    dump("runs", api.list_runs())
    reps = api.replays()
    dump("replays", reps)
    for r in reps:
        data = api.rp.load(r["key"])
        if data is not None:
            dump(f"replay_{r['key']}", data)
    if MEDIA.exists():
        shutil.rmtree(MEDIA)
    copied = []
    for rel in api.media_list():
        if rel.endswith(".mp4") and "--with-video" not in sys.argv:
            continue
        dst = MEDIA / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(api.REPORT / rel, dst)
        copied.append(rel)
    dump("media", copied)
    files = sorted(p.name for p in DATA.iterdir())
    print(f"wrote {len(files)} files to {DATA}: {', '.join(files)}")


if __name__ == "__main__":
    main()
