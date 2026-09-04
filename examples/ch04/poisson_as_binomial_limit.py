"""ポアソン分布は「n を大きく、p を小さく」した二項分布の行き着く先。

$\\lambda = np$ を 3 に固定したまま、$n = 10, 100, 10{,}000$ と増やし、
$p = \\lambda/n$ を同じだけ小さくする。平均は常に 3 のまま、分布の形だけが変わって
ポアソンに近づく。「めったに起きないことが、たくさんの機会にさらされる」——サーバの
エラー件数も、1日の問い合わせ件数も、この形になる。

近づき方は2つの目盛りで測る。全変動距離（分布そのものの距離）と、
$\\operatorname{Var}/\\mathbb{E}$（二項なら $1-p$、ポアソンなら常に 1）である。

    uv run python examples/ch04/poisson_as_binomial_limit.py
"""

import numpy as np
from scipy import stats

from toukei_tashikame import plots

LAM = 3.0
NS = (10, 100, 10_000)
DRAWS = 200_000
SEED = 43
SUPPORT = np.arange(0, 31)


def tv_distance(p: np.ndarray, q: np.ndarray) -> float:
    """全変動距離。2つの分布の差の絶対値を足して半分にする。"""
    return float(0.5 * np.abs(p - q).sum())


def main() -> None:
    rng = np.random.default_rng(SEED)
    pois = stats.poisson(mu=LAM)
    pois_pmf = pois.pmf(SUPPORT)

    print(f"--- λ = {LAM:g} を固定して n を増やす（各 {DRAWS:,} 回引く）---")
    print(f"{'n':>8}{'p = λ/n':>10}{'平均':>9}{'分散':>9}{'Var/E':>8}"
          f"{'TV(実測, Poi)':>14}{'TV(Bin, Poi)':>14}")

    emp_pmfs = []
    for n in NS:
        p = LAM / n
        x = rng.binomial(n, p, size=DRAWS)
        emp = np.bincount(x, minlength=SUPPORT.size)[: SUPPORT.size] / DRAWS
        emp_pmfs.append(emp)
        exact = stats.binom(n=n, p=p).pmf(SUPPORT)
        print(f"{n:>8,}{p:>10.4f}{x.mean():>9.3f}{x.var(ddof=1):>9.3f}"
              f"{x.var(ddof=1) / x.mean():>8.3f}"
              f"{tv_distance(emp, pois_pmf):>14.4f}{tv_distance(exact, pois_pmf):>14.4f}")

    print(f"\n  ポアソンの理論値          平均 {LAM:.3f}  分散 {LAM:.3f}  Var/E 1.000")
    print("  Var/E は二項なら 1−p にぴったり等しい。p が小さくなるほど 1 に寄る")
    print("  TV(Bin, Poi) は pmf を突き合わせた厳密値。n=10,000 では 0.0001 まで縮む")
    noise = tv_distance(emp_pmfs[-1], pois_pmf)
    print(f"  一方 TV(実測, Poi) は {noise:.4f} で下げ止まる。{DRAWS:,} 回の数え上げでは"
          f"これ以上の近さを見分けられない")
    print("  ← 数え上げには分解能がある。近さが標本誤差より小さくなったら、式で確かめる")

    plots.setup()
    fig, axes = plots.figure(1, 2, w=1.0, h=0.85, constrained_layout=True)

    ax = axes[0]
    width = 0.26
    for i, (n, emp) in enumerate(zip(NS, emp_pmfs, strict=True)):
        ax.bar(SUPPORT + (i - 1) * width, emp, width=width, lw=0,
               color=plots.PALETTE["data"], alpha=0.30 + 0.30 * i)
        ax.annotate(f"n={n:,}", xy=(9.4, 0.23 - 0.035 * i), fontsize=5.8,
                    color=plots.PALETTE["data"], alpha=0.45 + 0.25 * i)
    ax.plot(SUPPORT, pois_pmf, color=plots.PALETTE["truth"], lw=1.2, ls="--",
            dashes=(4, 2.0), zorder=5)
    ax.annotate("Poisson(3)", xy=(5.2, pois.pmf(5)), fontsize=6.0,
                color=plots.PALETTE["truth"], ha="left", va="bottom")
    ax.set_xlim(-0.7, 12)
    ax.set_xlabel("件数 k")
    ax.set_ylabel("確率")
    ax.set_title("n を増やすとポアソンに寄る")

    ax = axes[1]
    ns = np.array([2, 5, 10, 30, 100, 300, 1000, 3000, 10_000])
    tv = [tv_distance(stats.binom(n=int(n), p=LAM / n).pmf(SUPPORT), pois_pmf) for n in ns]
    ax.loglog(ns, tv, color=plots.PALETTE["estimate"], lw=1.2, marker="o", ms=2.5)
    ax.axhline(noise, color=plots.PALETTE["reject"], lw=0.9, ls="--", dashes=(4, 2.2))
    ax.annotate(f"{DRAWS:,}回の数え上げの分解能", xy=(3, noise), fontsize=5.8,
                color=plots.PALETTE["reject"], ha="left", va="bottom")
    ax.set_xlabel("n（λ=3 を固定）")
    ax.set_ylabel("全変動距離")
    ax.set_title("距離は 1/n に比例して縮む")

    plots.save(fig, "fig-4-3-poisson-limit.png")


if __name__ == "__main__":
    main()
