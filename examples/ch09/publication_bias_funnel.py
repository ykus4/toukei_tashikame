"""有意なものだけが世に出る世界では、公表された効果は系統的に真値より大きい。

出版バイアスは、誰かが数字をいじった結果ではない。**有意でなかった研究が引き出しに
しまわれる**というそれだけの足切りが、公表された文献の平均効果を系統的に押し上げる。
しかも小さな研究ほど有意になるには大きな効果が必要なので、押し上げ幅は n が小さい研究で
大きい。ファネルプロットの左下が空くのは、この非対称の直接の絵である。

真の効果 d=0.20 の研究を 2,000 本（研究のサンプルサイズ n は 20〜400 でばらつき、
半分ずつを2群に割る）シミュレートし、全研究の平均効果と、発表された研究だけの平均効果を
比べる。

    uv run python examples/ch09/publication_bias_funnel.py
"""

import itertools

import numpy as np
from scipy import special

from toukei_tashikame import plots, sim

D_TRUE = 0.20
N_MIN, N_MAX = 20, 400
ALPHA = 0.05
STUDIES = 2_000
SEED = 99


def one_study(rng) -> tuple[float, ...]:
    """1本の研究。規模 n を引いて2群に割り、効果量とその標準誤差、p 値を返す。"""
    n = int(rng.integers(N_MIN, N_MAX + 1))
    g = n // 2                                    # 群あたりの人数
    a = rng.normal(0.0, 1.0, size=g)
    b = rng.normal(D_TRUE, 1.0, size=g)

    sp = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    d_hat = (b.mean() - a.mean()) / sp

    s1, s2 = a.var(ddof=1) / g, b.var(ddof=1) / g
    t = (b.mean() - a.mean()) / np.sqrt(s1 + s2)
    df = (s1 + s2) ** 2 / (s1**2 / (g - 1) + s2**2 / (g - 1))
    p = float(2 * special.stdtr(df, -abs(t)))

    # d の標準誤差（Hedges の近似）。ファネルプロットの縦軸になる。
    se_d = np.sqrt(2.0 / g + d_hat**2 / (4 * g))
    return float(n), d_hat, se_d, p


def draw(n: np.ndarray, d: np.ndarray, se: np.ndarray, published: np.ndarray) -> None:
    fig, axes = plots.figure(1, 2, w=2.0, h=1.0)

    ax = axes[0]
    ax.scatter(d[~published], se[~published], s=4, color=plots.PALETTE["data"],
               alpha=0.30, lw=0, zorder=2)
    ax.scatter(d[published], se[published], s=4, color=plots.PALETTE["estimate"],
               alpha=0.75, lw=0, zorder=3)
    # 有意になる境界。|d| > 1.96 * se の外側だけが世に出る。
    grid = np.linspace(se.min(), se.max(), 50)
    for sign in (-1, 1):
        ax.plot(sign * 1.96 * grid, grid, color=plots.PALETTE["reject"], lw=0.9,
                ls="--", dashes=(4, 2.2), zorder=4)
    plots.mark_truth(ax, D_TRUE, f"真の効果 = {D_TRUE}")
    ax.invert_yaxis()
    ax.set_xlabel("推定された効果量 d")
    ax.set_ylabel("標準誤差（下ほど小さい研究）")
    ax.set_title("ファネルの左下が空く")
    ax.annotate("発表された研究", xy=(0.97, 0.06), xycoords="axes fraction", ha="right",
                fontsize=6.0, color=plots.PALETTE["estimate"])
    ax.annotate("引き出しの中", xy=(0.03, 0.06), xycoords="axes fraction",
                fontsize=6.0, color=plots.PALETTE["data"])

    ax = axes[1]
    edges = [20, 50, 100, 200, 300, 401]
    centers, all_mean, pub_mean = [], [], []
    for lo, hi in itertools.pairwise(edges):
        m = (n >= lo) & (n < hi)
        centers.append((lo + hi) / 2)
        all_mean.append(d[m].mean())
        pub_mean.append(d[m & published].mean() if (m & published).any() else np.nan)
    ax.plot(centers, pub_mean, color=plots.PALETTE["estimate"], lw=1.3, marker="o", ms=2.6)
    ax.plot(centers, all_mean, color=plots.PALETTE["data"], lw=1.1, marker="o", ms=2.2)
    plots.mark_truth(ax, D_TRUE, f"真の効果 = {D_TRUE}", axis="y")
    ax.annotate("発表された研究だけ", xy=(centers[1], pub_mean[1]), xytext=(4, 4),
                textcoords="offset points", fontsize=6.0, color=plots.PALETTE["estimate"])
    ax.annotate("全研究", xy=(centers[2], all_mean[2]), xytext=(4, -9),
                textcoords="offset points", fontsize=6.0, color=plots.PALETTE["data"])
    ax.set_xlabel("研究のサンプルサイズ n")
    ax.set_ylabel("効果量の平均")
    ax.set_ylim(0, max(pub_mean) * 1.2)
    ax.set_title("小さい研究ほど大きく盛られる")

    plots.save(fig, "fig-9-9-publication-bias-funnel.png")


def main() -> None:
    plots.setup()
    with sim.Timer("9-9 出版バイアス"):
        out = sim.repeat(one_study, trials=STUDIES, seed=SEED, progress=False)
    n, d, se, p = out.T
    published = p < ALPHA

    print(f"真の効果 d={D_TRUE}、研究 {STUDIES:,} 本、規模 n は "
          f"{N_MIN}〜{N_MAX} の一様（半分ずつ2群に割る）、"
          f"有意（p<{ALPHA}）なものだけが発表される\n")
    print(f"  全研究の平均効果          {d.mean():.4f}   （真値 {D_TRUE}）")
    print(f"  発表された研究だけの平均   {d[published].mean():.4f}"
          f"   （{d[published].mean() / D_TRUE:.1f} 倍）")
    print(f"  引き出しに残った研究の平均 {d[~published].mean():.4f}")
    print(f"  発表率                    {published.mean():.4f}"
          f"   （{int(published.sum()):,} / {STUDIES:,} 本）\n")

    print(f"{'研究の規模 n':>14}{'本数':>7}{'発表率':>9}{'全研究の平均 d':>16}"
          f"{'発表された平均 d':>18}")
    for lo, hi in ((20, 50), (50, 100), (100, 200), (200, 300), (300, 401)):
        m = (n >= lo) & (n < hi)
        pub = m & published
        label = f"{lo}〜{hi - 1}"
        print(f"{label:>14}{int(m.sum()):>7}{pub.mean() / m.mean():>9.4f}"
              f"{d[m].mean():>16.4f}{d[pub].mean():>18.4f}")

    small = (n < 50) & published
    large = (n >= 300) & published
    print(f"\n小規模な研究（n<50）が発表されるには d が {d[small].mean():.4f} 必要で"
          f"（真値の {d[small].mean() / D_TRUE:.1f} 倍）、")
    print(f"大規模な研究（n≥300）なら {d[large].mean():.4f} "
          f"（{d[large].mean() / D_TRUE:.1f} 倍）で足りる。")
    print("この非対称がファネルの左下を空にする。誰も不正をしていないのに、である。")
    draw(n, d, se, published)


if __name__ == "__main__":
    main()
