"""Tables + figures for the report from results/*.csv."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
RES = ROOT / "results"
FIG = ROOT / "report" / "figures"
BLUE, ORANGE, GREY = "#2a6fdb", "#e8833a", "#8a8f98"


def wilson(k, n, z=1.96):
    if n == 0:
        return 0, 0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return c - h, c + h


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    lines = []
    nav = RES / "e1_e4_nav.csv"
    if nav.exists():
        from eval_nav import summarize
        df = pd.read_csv(nav)
        s = summarize(df)
        s.to_csv(RES / "e1_e4_summary.csv")
        lines += ["## E1 / E4 navigation", s.to_markdown(), ""]
        e1 = df[df.safety]
        names = {1: "static", 2: "+ person", 3: "+ person + pet"}
        fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
        g = e1.groupby("level")
        rates = g.success.mean()
        ci = [wilson(int(g.success.sum()[l]), int(g.size()[l])) for l in rates.index]
        ax[0].bar([names[l] for l in rates.index], rates * 100, color=BLUE,
                  yerr=[[100 * (r - c[0]) for r, c in zip(rates, ci)], [100 * (c[1] - r) for r, c in zip(rates, ci)]],
                  capsize=4)
        ax[0].set_ylim(0, 105)
        ax[0].set_ylabel("reached goal (%)")
        ax[0].set_title("E1 success (95% Wilson CI)")
        lv = [2, 3]
        on = df[df.safety & df.level.isin(lv)].groupby("level")
        ax[1].bar([names[l] for l in lv], on.person_contacts.sum().reindex(lv), color=BLUE, label="person")
        ax[1].bar([names[l] for l in lv], on.pet_contacts.sum().reindex(lv),
                  bottom=on.person_contacts.sum().reindex(lv), color=ORANGE, label="pet")
        ax[1].set_title("E1 contact events (100 runs each)")
        ax[1].legend()
        e4 = df[(df.level == 2)]
        gg = e4.groupby("safety")
        lab = {True: "safety ON", False: "safety OFF"}
        ax[2].bar([lab[k] for k in gg.size().index], gg.person_contacts.sum(), color=[ORANGE, BLUE])
        ax[2].set_title("E4: person contacts, level 2")
        for a in ax:
            a.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(FIG / "e1_e4.png", dpi=150)
        fig, ax = plt.subplots(figsize=(6, 3.4))
        for k, c in ((True, BLUE), (False, ORANGE)):
            d = e4[e4.safety == k].min_person_dist.clip(upper=2.0)
            ax.hist(d, bins=30, alpha=0.6, color=c, label=lab[k])
        ax.axvline(0, color="k", lw=1)
        ax.set_xlabel("min robot-person surface distance per run (m)")
        ax.set_ylabel("runs")
        ax.legend()
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(FIG / "e4_min_dist.png", dpi=150)
    for f in sorted(RES.glob("e2_manip_*.csv")):
        df = pd.read_csv(f)
        t = df.groupby(["policy", "skill"]).success.agg(["count", "mean"])
        lines += [f"## E2 manipulation ({f.name})", t.to_markdown(), ""]
    for f in sorted(RES.glob("e3_full_*.csv")):
        df = pd.read_csv(f)
        lines += [f"## E3 full task ({f.name})",
                  f"success {df.success.sum()}/{len(df)} = {df.success.mean():.0%}; "
                  f"person contacts {df.person_contacts.sum()} (robot moving {df.person_contacts_robot_moving.sum()})",
                  df[~df.success].failed_at.value_counts().to_markdown(), ""]
        fig, ax = plt.subplots(figsize=(6, 3))
        vc = df.failed_at.replace("", "SUCCESS").value_counts()
        ax.barh(vc.index, vc.values, color=[BLUE if i == "SUCCESS" else ORANGE for i in vc.index])
        ax.set_title(f"E3 outcomes ({len(df)} runs)")
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout()
        fig.savefig(FIG / "e3_outcomes.png", dpi=150)
    (RES / "summary.md").write_text("\n".join(lines), encoding="utf8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
