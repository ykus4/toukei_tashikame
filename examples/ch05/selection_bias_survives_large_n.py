"""選択バイアスは n を増やしても消えない — 標本の「大きさ」と「代表性」は別物である。

満足度 y ~ N(50, 10²) の母集団にアンケートを配る。ただし回答するのは満足度が上位50%
（=母集団の中央値より上）の人だけ。この標本で平均を推定すると、n をいくら増やしても
57.98 前後に張りつく。真値は 50 なのに、である。

大数の法則が保証するのは「同じ分布から独立に引いた標本の平均が、その分布の平均に
収束する」ことでしかない。回答者は母集団から引かれていない——切り取られた別の分布から
引かれている。n を増やして縮むのは標準誤差だけで、区間は間違った場所で細くなる。
**n が大きいほど、自信を持って間違える。**

    uv run python examples/ch05/selection_bias_survives_large_n.py
"""

import unicodedata

import numpy as np

from toukei_tashikame import estimate, plots, sim

MU, SIGMA = 50.0, 10.0
N_LIST = [100, 1_000, 10_000, 100_000]
REPEATS = 400
SEED = 24
# 上位50%だけが回答するときの理論値: E[y | y > μ] = μ + σ·φ(0)/0.5
BIASED_MEAN = MU + SIGMA * np.exp(0.0) / np.sqrt(2 * np.pi) / 0.5


def survey(rng, n: int, *, biased: bool) -> np.ndarray:
    """アンケートの回答者を返す。``biased`` なら満足度上位50%だけが回答する。"""
    if not biased:
        return rng.normal(MU, SIGMA, size=n)
    invited = rng.normal(MU, SIGMA, size=2 * n)     # 回答率50%を見込んで2n人に配る
    return invited[invited > MU]                    # 母集団の中央値より上だけが答える


def rj(text: str, width: int) -> str:
    """全角を2桁として数えて右詰めする。日本語の見出しでも表の桁が揃う。"""
    w = sum(0 if unicodedata.combining(c) else
            2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in text)
    return " " * max(width - w, 0) + text


def main() -> None:
    plots.setup()
    print(f"--- 母集団 N({MU:g}, {SIGMA:g}²)、回答するのは満足度が上位50%の人だけ ---")
    print(f"  回答者の分布の平均（理論）= {BIASED_MEAN:.4f}   真値との差 "
          f"{BIASED_MEAN - MU:+.4f}\n")
    print("  " + rj("n", 8) + "  " + rj("推定値", 9) + "  " + rj("ずれ", 8)
          + "  " + rj("区間幅", 8) + "  " + rj("被覆", 7)
          + "  " + rj("参考: 無作為の被覆", 20))

    rows = []
    for i, n in enumerate(N_LIST):
        def one(rng, n=n, biased=True):
            return estimate.ci_mean_t(survey(rng, n, biased=biased))

        biased = sim.coverage(one, truth=MU, trials=REPEATS, seed=SEED + i,
                              progress=False)
        fair = sim.coverage(lambda rng, n=n: estimate.ci_mean_t(survey(rng, n, biased=False)),
                            truth=MU, trials=REPEATS, seed=SEED + i, progress=False)
        centers = biased.intervals.mean(axis=1)
        width = float(np.mean(biased.intervals[:, 1] - biased.intervals[:, 0]))
        rows.append((n, float(centers.mean()), width, biased.rate, fair.rate))
        print(f"  {n:>8,}  {centers.mean():>9.2f}  {centers.mean() - MU:>+8.2f}"
              f"  {width:>8.3f}  {biased.rate:>7.3f}  {fair.rate:>20.3f}")

    first, last = rows[0], rows[-1]
    print(f"\n  n を {first[0]:,} から {last[0]:,} へ 1,000倍にしても、"
          f"推定値は {first[1]:.2f} → {last[1]:.2f}。真値 {MU:g} には近づかない")
    print(f"  縮んだのは区間の幅だけ: {first[2]:.3f} → {last[2]:.3f}"
          f"（{first[2] / last[2]:.0f}分の1）")
    print(f"  真値からのずれは SE の {abs(last[1] - MU) / (last[2] / 3.92):.0f} 倍。"
          "無作為抽出なら同じ n で被覆は 0.95 前後に収まる（右端の列）")
    print("  n を増やして減らせるのは偶然のばらつきだけで、バイアスは減らせない")

    fig, (ax1, ax2) = plots.figure(1, 2, w=2.0)
    pop = np.random.default_rng(SEED).normal(MU, SIGMA, size=200_000)
    ax1.hist(pop, bins=70, density=True, color=plots.PALETTE["data"], alpha=0.45, lw=0)
    ax1.hist(pop[pop > MU], bins=35, density=True, color=plots.PALETTE["reject"],
             alpha=0.55, lw=0)
    plots.mark_truth(ax1, MU, f"母平均 = {MU:g}")
    ax1.axvline(BIASED_MEAN, color=plots.PALETTE["estimate"], lw=1.0, ls="--",
                dashes=(4, 2.0))
    ax1.annotate(f"回答者の平均 = {BIASED_MEAN:.2f}", xy=(BIASED_MEAN, 0.80),
                 xycoords=("data", "axes fraction"), xytext=(3, 0),
                 textcoords="offset points", fontsize=6.0,
                 color=plots.PALETTE["estimate"])
    ax1.set_title("回答するのはオレンジの人だけ")
    ax1.set_xlabel("満足度")
    ax1.set_ylabel("密度")

    ns = np.array([r[0] for r in rows], dtype=float)
    est = np.array([r[1] for r in rows])
    half = np.array([r[2] for r in rows]) / 2
    ax2.errorbar(ns, est, yerr=half, fmt="o-", ms=3, lw=1.0,
                 color=plots.PALETTE["estimate"], ecolor=plots.PALETTE["estimate"],
                 capsize=2)
    plots.mark_truth(ax2, MU, f"真値 = {MU:g}", axis="y")
    ax2.set_xscale("log")
    ax2.set_ylim(MU - 2, BIASED_MEAN + 2)
    ax2.set_xlabel("回答者数 n（対数）")
    ax2.set_ylabel("推定値と95%区間")
    ax2.set_title("区間は縮む。ただし間違った場所で")
    plots.save(fig, "fig-5-8-selection-bias.png")


if __name__ == "__main__":
    main()
