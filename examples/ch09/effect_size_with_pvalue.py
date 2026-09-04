"""p 値だけでは何も分からない。効果量を並べて初めて「どれくらい」が言える。

p 値は「効果があるか」ではなく「効果ゼロのもとでこのデータが珍しいか」を測る。n を
増やせばどんな小さな効果でも有意になり、n が小さければ大きな効果でも有意にならない。
だから報告には効果量が要る。

真の効果 d=0.3 のデータで Cohen's d・オッズ比・相関比（$\\eta^2$）を手で書き、
n=20/100/1000 での推定分布と p 値の関係を 5,000 回ずつ見る。効果量そのものも推定値
であって、n が小さければ盛大にばらつく——それも同時に数える。

    uv run python examples/ch09/effect_size_with_pvalue.py
"""

import numpy as np
from scipy import special

from toukei_tashikame import plots, sim

D_TRUE = 0.3
NS = (20, 100, 1000)
ALPHA = 0.05
TRIALS = 5_000
SEED = 97


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """プールした標準偏差で標準化した平均差。手で書く。"""
    n1, n2 = a.size, b.size
    sp = np.sqrt(((n1 - 1) * a.var(ddof=1) + (n2 - 1) * b.var(ddof=1)) / (n1 + n2 - 2))
    return float((b.mean() - a.mean()) / sp)


def odds_ratio(a: np.ndarray, b: np.ndarray) -> float:
    """全体の中央値で二値化した 2×2 表のオッズ比。Haldane の 0.5 補正つき。"""
    cut = np.median(np.concatenate([a, b]))
    n11, n10 = float((b > cut).sum()) + 0.5, float((b <= cut).sum()) + 0.5
    n01, n00 = float((a > cut).sum()) + 0.5, float((a <= cut).sum()) + 0.5
    return float((n11 * n00) / (n10 * n01))


def eta_squared(a: np.ndarray, b: np.ndarray) -> float:
    """相関比 $\\eta^2$。群で説明できる分散の割合（= 点双列相関の2乗）。"""
    y = np.concatenate([a, b])
    grand = y.mean()
    ss_between = a.size * (a.mean() - grand) ** 2 + b.size * (b.mean() - grand) ** 2
    ss_total = float(((y - grand) ** 2).sum())
    return float(ss_between / ss_total) if ss_total > 0 else 0.0


def make_trial(n: int):
    def one_trial(rng) -> tuple[float, ...]:
        a = rng.normal(0.0, 1.0, size=n)
        b = rng.normal(D_TRUE, 1.0, size=n)
        s1, s2 = a.var(ddof=1) / n, b.var(ddof=1) / n
        t = (b.mean() - a.mean()) / np.sqrt(s1 + s2)
        df = (s1 + s2) ** 2 / (s1**2 / (n - 1) + s2**2 / (n - 1))
        p = float(2 * special.stdtr(df, -abs(t)))
        return cohens_d(a, b), odds_ratio(a, b), eta_squared(a, b), p

    return one_trial


def draw(results: dict[int, np.ndarray]) -> None:
    fig, axes = plots.figure(1, 2, w=2.0, h=0.95)

    ax = axes[0]
    shades = ["#cfe0f0", "#7fb3e0", plots.PALETTE["estimate"]]
    bins = np.linspace(-1.2, 1.6, 60)
    for n, color in zip(NS, shades, strict=True):
        ax.hist(results[n][:, 0], bins=bins, density=True, color=color, alpha=0.65,
                lw=0, label=f"n={n}")
    plots.mark_truth(ax, D_TRUE, f"真の d = {D_TRUE}")
    ax.set_xlabel("推定された Cohen's d")
    ax.set_ylabel("密度")
    ax.legend(loc="upper left")
    ax.set_title("効果量そのものも推定値でばらつく")

    ax = axes[1]
    d, p = results[100][:, 0], results[100][:, 3]
    sig = p < ALPHA
    ax.scatter(d[~sig], p[~sig], s=2.0, color=plots.PALETTE["data"], alpha=0.35, lw=0)
    ax.scatter(d[sig], p[sig], s=2.0, color=plots.PALETTE["reject"], alpha=0.5, lw=0)
    ax.axhline(ALPHA, color=plots.PALETTE["reject"], lw=0.9, ls="--", dashes=(4, 2.2))
    ax.annotate("α = 0.05", xy=(0.99, ALPHA), xycoords=("axes fraction", "data"),
                xytext=(-2, 3), textcoords="offset points", ha="right", fontsize=6.0,
                color=plots.PALETTE["reject"])
    plots.mark_truth(ax, D_TRUE, f"真の d = {D_TRUE}")
    ax.set_yscale("log")
    ax.set_xlabel("推定された Cohen's d（n=100）")
    ax.set_ylabel("p 値（対数目盛）")
    ax.set_title("p と d は同じ量の裏表でしかない")

    plots.save(fig, "fig-9-7-effect-size-vs-pvalue.png")


def main() -> None:
    plots.setup()
    results: dict[int, np.ndarray] = {}
    with sim.Timer("9-7 効果量とp値"):
        for i, n in enumerate(NS):
            results[n] = sim.repeat(make_trial(n), trials=TRIALS, seed=SEED + 10 * i,
                                    progress=False)

    print(f"真の効果 d={D_TRUE}、各 n で {TRIALS:,} 回\n")
    print(f"{'n':>6}{'d の平均':>11}{'d の SD':>10}{'オッズ比の中央値':>18}"
          f"{'η² の平均':>11}{'検出力':>9}{'corr(d, log p)':>16}")
    for n in NS:
        d, or_, eta2, p = results[n].T
        # p は 0 に張りつくので対数で見る。d と p は同じ量の別表現でしかない。
        corr = float(np.corrcoef(d, np.log10(np.maximum(p, 1e-300)))[0, 1])
        print(f"{n:>6}{d.mean():>11.4f}{d.std(ddof=1):>10.4f}"
              f"{np.median(or_):>18.4f}{eta2.mean():>11.4f}"
              f"{(p < ALPHA).mean():>9.4f}{corr:>16.4f}")

    print(f"\nd の推定は n=20 で SD {results[20][:, 0].std(ddof=1):.4f}"
          f"（真値 {D_TRUE} と同じ桁のばらつき）、"
          f"n=1000 で {results[1000][:, 0].std(ddof=1):.4f} まで縮む。")
    print("平均はどの n でも真値の近くにある。動いているのは幅だけである。\n")

    # 3つの効果量は同じものを別の目盛で言っている（変換式で行き来できる）。
    d_bar = float(results[1000][:, 0].mean())
    print("同じ効果を3つの目盛で言い直す（n=1000 の平均）:")
    print(f"  Cohen's d      {d_bar:.4f}   （真値 {D_TRUE}）")
    print(f"  オッズ比        {np.median(results[1000][:, 1]):.4f}"
          f"   （目安 $\\exp(d\\pi/\\sqrt3)$ = "
          f"{np.exp(D_TRUE * np.pi / np.sqrt(3)):.4f}。中央値で二値化した"
          f"ぶん情報が落ちて小さめに出る）")
    print(f"  相関比 η²       {results[1000][:, 2].mean():.4f}"
          f"   （$d^2/(d^2+4)$ = {D_TRUE**2 / (D_TRUE**2 + 4):.4f}）")

    n20, n1000 = (results[20][:, 3] < ALPHA), (results[1000][:, 3] < ALPHA)
    print(f"\n同じ真の効果でも、有意になる割合は n=20 で {n20.mean():.4f}、"
          f"n=1000 で {n1000.mean():.4f}。")
    print(f"しかも n=20 で有意になった試行の d の平均は "
          f"{results[20][n20, 0].mean():.4f} で、真値の "
          f"{results[20][n20, 0].mean() / D_TRUE:.1f} 倍に膨らむ（勝者の呪い）。")
    print("p 値だけを報告すると、この2つの事実がどちらも見えなくなる。")
    draw(results)


if __name__ == "__main__":
    main()
