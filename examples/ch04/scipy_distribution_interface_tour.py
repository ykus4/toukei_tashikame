"""分布はどれも同じ4つのメソッドで扱える — rvs / pdf(pmf) / cdf / ppf。

scipy.stats の分布オブジェクトは、連続でも離散でも同じ名前のメソッドを持つ。
覚えるのは「引く（rvs）」「高さ（pdf/pmf）」「左側の面積（cdf）」「面積から戻す（ppf）」
の4つだけで、以降の章で出てくる分布はすべてこの窓から触る。

ここでは cdf と ppf が互いの逆であること（連続なら誤差ゼロで戻る）、そして rvs で
引いた100,000点の経験CDFが理論の cdf にどこまで寄るかを数える。離散分布では ppf が
階段の逆になるので、往復が片方向にしか成り立たないことも一緒に見る。

    uv run python examples/ch04/scipy_distribution_interface_tour.py
"""

import numpy as np
from scipy import stats

from toukei_tashikame import plots

SEED = 41
N = 100_000

# 連続1つ・離散2つ。同じメソッド名で同じように扱えることが要点。
DISTS = [
    ("norm(0, 1)", stats.norm(loc=0.0, scale=1.0), False),
    ("binom(10, 0.3)", stats.binom(n=10, p=0.3), True),
    ("poisson(3)", stats.poisson(mu=3.0), True),
]


def ecdf_max_gap(x: np.ndarray, dist, discrete: bool) -> float:
    """経験CDFと理論CDFの最大差。KS統計量そのものだが、ここでは手で数える。

    経験CDFは階段なので、段の上と下の両方で比べる。離散分布は理論のCDFも同じ点で
    跳ぶため、段の上（値そのもの）だけを見ればよい。
    """
    values, counts = np.unique(x, return_counts=True)
    upper = np.cumsum(counts) / x.size       # その値までの経験CDF（段の上）
    theory = dist.cdf(values)
    if discrete:
        return float(np.max(np.abs(upper - theory)))
    lower = upper - counts / x.size          # その値の直前（段の下）
    return float(max(np.max(upper - theory), np.max(theory - lower)))


def four_methods_table(rng) -> None:
    print("--- 1. 4つのメソッドを同じ形で叩く ---")
    print(f"{'分布':<16}{'rvs の先頭3つ':<26}{'pdf/pmf(x0)':>12}{'cdf(x0)':>10}{'ppf(0.975)':>12}")
    for name, dist, discrete in DISTS:
        draws = dist.rvs(size=3, random_state=rng)
        x0 = float(dist.median())
        height = dist.pmf(x0) if discrete else dist.pdf(x0)
        head = np.array2string(np.asarray(draws), precision=3, suppress_small=True)
        print(f"{name:<16}{head:<26}{height:>12.4f}{dist.cdf(x0):>10.4f}{dist.ppf(0.975):>12.4f}")
    print("  x0 は各分布の中央値。離散は pdf ではなく pmf だが、それ以外の名前は共通")


def round_trip() -> None:
    print("\n--- 2. cdf と ppf は互いの逆 ---")
    z = 1.96
    back = stats.norm.ppf(stats.norm.cdf(z))
    print(f"  連続: ppf(cdf({z})) - {z} = {back - z:.12f}")
    print(f"  連続: cdf(ppf(0.975)) - 0.975 = {stats.norm.cdf(stats.norm.ppf(0.975)) - 0.975:.12f}")

    pois = stats.poisson(mu=3.0)
    k = 4
    print(f"  離散: ppf(cdf({k})) = {pois.ppf(pois.cdf(k)):.0f}   ← 整数には戻る")
    u = 0.60
    print(f"  離散: cdf(ppf({u})) = {pois.cdf(pois.ppf(u)):.4f}   ← {u} には戻らない")
    print("  離散のCDFは階段なので、面積 0.60 ちょうどに対応する点が存在しない。"
          "ppf は「その面積を超える最小の整数」を返す")


def empirical_vs_theory(rng) -> None:
    print(f"\n--- 3. rvs で引いた {N:,} 点の経験CDF vs 理論CDF ---")
    for name, dist, discrete in DISTS:
        x = dist.rvs(size=N, random_state=rng)
        gap = ecdf_max_gap(x, dist, discrete)
        print(f"  {name:<16}最大差 {gap:.4f}   （目安 1/√n = {1 / np.sqrt(N):.4f}）")
    print("  差は 1/√n の程度に収まる。標本を増やせば経験分布は理論分布に寄る"
          "（第5章で速さを測る）")


def make_figure() -> None:
    plots.setup()
    fig, axes = plots.figure(2, 2, h=1.6, constrained_layout=True)
    dist = stats.norm(loc=0.0, scale=1.0)
    grid = np.linspace(-3.5, 3.5, 400)
    rng = np.random.default_rng(SEED)

    ax = axes[0, 0]
    ax.plot(grid, dist.pdf(grid), color=plots.PALETTE["truth"], lw=1.2)
    ax.set_title("pdf — 高さ")
    ax.set_xlabel("x")

    ax = axes[0, 1]
    ax.plot(grid, dist.cdf(grid), color=plots.PALETTE["truth"], lw=1.2)
    ax.set_title("cdf — 左側の面積")
    ax.set_xlabel("x")

    ax = axes[1, 0]
    q = np.linspace(0.001, 0.999, 400)
    ax.plot(q, dist.ppf(q), color=plots.PALETTE["truth"], lw=1.2)
    ax.set_title("ppf — 面積から x へ戻す")
    ax.set_xlabel("確率")

    ax = axes[1, 1]
    plots.sim_hist(ax, dist.rvs(size=N, random_state=rng), theory=(grid, dist.pdf(grid)),
                   bins=60, theory_label="pdf")
    ax.set_title(f"rvs — {N:,} 点を引く")
    ax.set_xlabel("x")

    plots.save(fig, "fig-4-1-four-methods.png")


def main() -> None:
    rng = np.random.default_rng(SEED)
    four_methods_table(rng)
    round_trip()
    empirical_vs_theory(rng)
    make_figure()


if __name__ == "__main__":
    main()
