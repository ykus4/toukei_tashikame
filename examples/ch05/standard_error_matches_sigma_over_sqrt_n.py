"""標準誤差 — 標本平均のばらつきは σ/√n ちょうどに縮む。

n を 5, 10, 25, 50, 100, 500 と変えて、各 n で10,000回ずつ標本を引き、標本平均の
実測SDと σ/√n を突き合わせる。この式が言っているのは「n を4倍にしないとSEは半分に
ならない」ということで、精度が n ではなく √n でしか買えないという事実は、
サンプルサイズ設計の話（第10章）の出発点になる。

標準偏差 σ と標準誤差 σ/√n は別物である。前者はデータのばらつき、後者は
**推定値のばらつき**。n を増やしても σ は縮まないが、σ/√n は縮む。

    uv run python examples/ch05/standard_error_matches_sigma_over_sqrt_n.py
"""

import unicodedata
from itertools import pairwise

import numpy as np

from toukei_tashikame import plots, sim

MU, SIGMA = 50.0, 10.0
N_LIST = [5, 10, 25, 50, 100, 500]
TRIALS = 10_000


def sampling_sd(n: int, seed: int) -> float:
    """n の標本を TRIALS 回引いて、標本平均の実測SDを返す。"""
    means = sim.repeat(lambda rng: float(rng.normal(MU, SIGMA, size=n).mean()),
                       trials=TRIALS, seed=seed, progress=False)
    return float(means.std(ddof=1))


def rj(text: str, width: int) -> str:
    """全角を2桁として数えて右詰めする。日本語の見出しでも表の桁が揃う。"""
    w = sum(0 if unicodedata.combining(c) else
            2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in text)
    return " " * max(width - w, 0) + text


def main() -> None:
    plots.setup()
    rows = []
    for i, n in enumerate(N_LIST):
        observed = sampling_sd(n, seed=100 + i)
        theory = SIGMA / np.sqrt(n)
        rows.append((n, observed, theory, observed / theory - 1.0))

    print(f"--- 標本平均のSD（各 n で {TRIALS:,}回）---")
    print("  " + rj("n", 5) + "  " + rj("実測SD", 8) + "  " + rj("σ/√n", 8)
          + "  " + rj("相対誤差", 8))
    for n, obs, th, rel in rows:
        print(f"  {n:>5}  {obs:>8.4f}  {th:>8.4f}  {100 * rel:>7.1f}%")
    worst = max(rows, key=lambda r: abs(r[3]))
    print(f"  最大相対誤差 {100 * abs(worst[3]):.1f}%（n={worst[0]}）"
          "   ← 10,000回のシミュレーション誤差の範囲")

    print("\n--- n を増やすと、SEは何倍になるか ---")
    print("  " + rj("n", 5) + " → " + rj("n2", 5) + "   " + rj("n の倍率", 7)
          + "   " + rj("実測の比", 8) + "  " + rj("理論 1/√倍率", 11))
    for (n, obs_n, _, _), (m, obs_m, _, _) in pairwise(rows):
        mult = m / n
        print(f"  {n:>5} → {m:>5}   {mult:>7.1f}   {obs_m / obs_n:>8.3f}  {1 / np.sqrt(mult):>11.3f}")
    ratio_4x = rows[4][1] / rows[2][1]      # n=25 → n=100 がちょうど4倍
    print(f"  n を4倍（25→100）にしたときのSEの比 = {ratio_4x:.3f}（理論 0.500）")
    print("  ← 精度は n ではなく √n でしか買えない。半分にしたければ4倍集める")

    ns = np.array([r[0] for r in rows], dtype=float)
    obs = np.array([r[1] for r in rows])
    fig, ax = plots.figure(w=1.15)
    fine = np.logspace(np.log10(4), np.log10(700), 200)
    ax.plot(fine, SIGMA / np.sqrt(fine), color=plots.PALETTE["truth"], lw=1.1, zorder=4)
    ax.annotate("理論 σ/√n", xy=(fine[120], SIGMA / np.sqrt(fine[120])), xytext=(4, 4),
                textcoords="offset points", fontsize=6.0, color=plots.PALETTE["truth"])
    ax.scatter(ns, obs, s=18, color=plots.PALETTE["estimate"], zorder=5)
    ax.annotate(f"実測（各 {TRIALS:,}回）", xy=(ns[1], obs[1]), xytext=(6, -10),
                textcoords="offset points", fontsize=6.0, color=plots.PALETTE["estimate"])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("標本サイズ n（対数）")
    ax.set_ylabel("標本平均のSD（対数）")
    ax.set_title("標準誤差は σ/√n — 両対数で傾き −1/2 の直線")
    plots.save(fig, "fig-5-3-standard-error.png")


if __name__ == "__main__":
    main()
