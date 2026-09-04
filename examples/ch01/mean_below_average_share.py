"""「平均年収」以下の人が半分を超えるのはなぜか。

平均が「真ん中」だと思っていると、平均以下が半分だと期待してしまう。右に裾を引く分布
では、少数の高い値が平均を引き上げるので、平均以下の人のほうが多数派になる。

対数正規（μ=6.0、σ=0.8）から 10,000 人を引いて数え上げる。理論値も出せる分布なので、
数え上げた割合を Φ(σ/2) と突き合わせておく。

    uv run python examples/ch01/mean_below_average_share.py
"""

import numpy as np
from scipy import stats

from toukei_tashikame import datasets, describe, plots

MU, SIGMA, N = 6.0, 0.8, 10_000


def main() -> None:
    plots.setup()

    # datasets.income は中央値を直接指定する。対数正規の中央値は exp(μ)。
    x = datasets.income(n=N, median=float(np.exp(MU)), sigma=SIGMA, seed=1)

    m = describe.mean(x)
    med = describe.median(x)
    share = float((x <= m).mean())

    # 対数正規なら理論値が閉じた形で出る。
    theory_mean = float(np.exp(MU + SIGMA**2 / 2))
    theory_median = float(np.exp(MU))
    theory_share = float(stats.norm.cdf(SIGMA / 2))

    print(f"--- 対数正規 μ={MU}, σ={SIGMA}, n={N:,} ---")
    print(f"{'':<18}{'標本':>12}{'理論値':>12}")
    print(f"{'平均':<18}{m:>12.1f}{theory_mean:>12.1f}")
    print(f"{'中央値':<18}{med:>12.1f}{theory_median:>12.1f}")
    print(f"{'平均以下の割合':<18}{share:>12.4f}{theory_share:>12.4f}")
    print(f"\n  平均は中央値より {m - med:.1f} 上（{100 * (m / med - 1):.1f}% 高い）")
    print(f"  平均以下は {int((x <= m).sum()):,} 人 / {N:,} 人。"
          f"「平均年収」を聞いて自分は下だと感じる人が多数派になる")
    print(f"  理論値 Φ(σ/2) = Φ({SIGMA / 2}) = {theory_share:.4f}。"
          "σ が大きいほど平均以下の割合は増える")

    fig, ax = plots.figure()
    ax.hist(x, bins=60, range=(0, 2000), color=plots.PALETTE["data"], alpha=0.55, lw=0)
    counts, edges = np.histogram(x, bins=60, range=(0, 2000))
    below = edges[:-1] < m
    ax.bar(edges[:-1][below], counts[below], width=np.diff(edges)[below], align="edge",
           color=plots.PALETTE["reject"], alpha=0.45, lw=0)
    ax.axvline(m, color=plots.PALETTE["estimate"], lw=1.2)
    ax.axvline(med, color=plots.PALETTE["alt"], lw=1.2, ls="--", dashes=(4, 2.0))
    ax.annotate(f"平均 {m:.0f}", xy=(m, 0.96), xycoords=("data", "axes fraction"),
                fontsize=6.0, color=plots.PALETTE["estimate"], ha="left", va="top",
                xytext=(3, 0), textcoords="offset points")
    ax.annotate(f"中央値 {med:.0f}", xy=(med, 0.80), xycoords=("data", "axes fraction"),
                fontsize=6.0, color=plots.PALETTE["alt"], ha="right", va="top",
                xytext=(-3, 0), textcoords="offset points")
    ax.annotate(f"平均以下 {share:.1%}", xy=(0.20, 0.55), xycoords="axes fraction",
                fontsize=6.4, color=plots.PALETTE["reject"], ha="center")
    ax.set_xlabel("年収（万円）")
    ax.set_ylabel("人数")
    ax.set_title("右に裾を引く分布では、平均以下が多数派になる")

    plots.save(fig, "fig-1-2-income-mean-vs-median.png")


if __name__ == "__main__":
    main()
