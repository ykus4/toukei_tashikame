"""ECDF にはビン幅がない。同じデータから分位点をそのまま読む。

ヒストグラムは「どう切るか」を決めないと描けないが、経験累積分布関数（ECDF）は
データ点をそのまま階段にするだけで、こちらが選ぶものが1つもない。中央値も第90
百分位も図から直読でき、正規分布との食い違いも縦の隙間として目で測れる。
その隙間の最大値が、そのまま KS 統計量である。

    uv run python examples/ch02/ecdf_beats_histogram_binwidth.py
"""

import numpy as np
from scipy import stats

from toukei_tashikame import datasets, plots


def ecdf(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(昇順の値, 累積割合)。i 番目の点までに全体の i/n が入る、という階段。"""
    xs = np.sort(np.asarray(x, dtype=float))
    return xs, np.arange(1, xs.size + 1) / xs.size


def read_at(xs: np.ndarray, ys: np.ndarray, q: float) -> float:
    """ECDF が初めて ``q`` 以上になる x。図から目で読むのと同じ操作。"""
    return float(xs[np.searchsorted(ys, q, side="left")])


def main() -> None:
    plots.setup()
    print("--- 2-5 ECDF はビン幅を選ばない ---")

    df = datasets.penguins()
    x = df["body_mass_g"].to_numpy(dtype=float)
    xs, ys = ecdf(x)
    print(f"Palmer Penguins の体重 body_mass_g  n={x.size}"
          f"  範囲 {x.min():.0f}–{x.max():.0f}g")

    print("\n[1] ECDF から分位点を直読する（階段が q を跨ぐ x を拾うだけ）")
    print(f"{'q':>6}{'ECDF から直読':>16}{'np.quantile(linear)':>22}")
    for q in (0.25, 0.50, 0.75, 0.90, 0.95):
        print(f"{q:>6.2f}{read_at(xs, ys, q):>14.1f}g{np.quantile(x, q):>20.1f}g")
    print(f"  中央値 {read_at(xs, ys, 0.50):.1f}g / 第90百分位 "
          f"{read_at(xs, ys, 0.90):.1f}g  ← 図から読める値")
    same = all(abs(read_at(xs, ys, q) - np.quantile(x, q)) < 1e-9
               for q in (0.25, 0.50, 0.75, 0.90, 0.95))
    print(f"  直読と linear 補間が全部一致したか: {same}")
    print(f"  ← 体重は 25g 刻みで記録されていて、342羽が {np.unique(x).size} 通りの値"
          "しか取らない。\n    階段の1段が厚いので、補間しても実在の観測点に落ちる")

    print("\n[2] 正規分布を当てはめて、階段との隙間を測る")
    mu, sd = x.mean(), x.std(ddof=1)
    print(f"  当てはめ N(μ={mu:.2f}, σ={sd:.2f})")
    ks = stats.kstest(x, stats.norm(mu, sd).cdf)
    print(f"  KS 距離 D={ks.statistic:.4f}（scipy の素の p={ks.pvalue:.4f}）")

    # 手で数えても同じ値が出る。D は階段と曲線の縦の隙間の最大値でしかない。
    cdf = stats.norm.cdf(xs, mu, sd)
    d_plus = float(np.max(ys - cdf))
    d_minus = float(np.max(cdf - (np.arange(xs.size) / xs.size)))
    print(f"  手計算 D+={d_plus:.4f} / D-={d_minus:.4f} → D={max(d_plus, d_minus):.4f}"
          f"（scipy との差 {abs(max(d_plus, d_minus) - ks.statistic):.1e}）")
    at = xs[int(np.argmax(ys - cdf))] if d_plus >= d_minus else xs[int(np.argmax(cdf - ys))]
    print(f"  隙間が最大になるのは {at:.0f}g のあたり")

    # μ と σ を同じデータから推定しているので、scipy の p はそのままでは使えない。
    from statsmodels.stats.diagnostic import lilliefors

    d_lf, p_lf = lilliefors(x, dist="norm", pvalmethod="table")
    print("\n  同じ D でも、μ・σ を当てはめたぶん帰無分布が変わる:")
    print(f"    scipy.kstest（μ・σ を既知として扱う）  p={ks.pvalue:.4f}")
    print(f"    Lilliefors 補正（推定したことを織り込む） D={d_lf:.4f}, p={p_lf:.4f}")
    print("  ← 素の KS は保守的（p を大きめに出す）。同じデータで当てはめてから"
          "\n    検定するなら補正版を使う。今回はどちらも 0.05 を下回るので結論は"
          "\n    変わらないが、境界付近では判定が入れ替わる")

    print("\n[3] ヒストグラムはビン幅次第で印象が変わる")
    for k in (8, 15, 30, 60):
        counts, edges = np.histogram(x, bins=k)
        padded = np.concatenate([[0], counts, [0]])
        peaks = int(((padded[1:-1] > padded[:-2]) & (padded[1:-1] >= padded[2:])).sum())
        print(f"    {k:>2}ビン（幅 {edges[1] - edges[0]:>6.1f}g）峰 {peaks} 個")
    print("  ← ECDF は上のどれとも無関係に、常に同じ1本の階段である")

    fig, axes = plots.figure(1, 2, h=1.05, w=2.0)
    ax = axes[0]
    ax.step(xs, ys, where="post", color=plots.PALETTE["data"], lw=1.1, zorder=3)
    grid = np.linspace(x.min() - 200, x.max() + 200, 400)
    ax.plot(grid, stats.norm.cdf(grid, mu, sd), color=plots.PALETTE["truth"],
            lw=1.0, ls="--", dashes=(4, 2), zorder=4)
    ax.annotate(f"当てはめた正規 N({mu:.0f}, {sd:.0f}²)", xy=(0.03, 0.90),
                xycoords="axes fraction", fontsize=6.0, color=plots.PALETTE["truth"])
    for q, name in ((0.50, "中央値"), (0.90, "第90百分位")):
        v = read_at(xs, ys, q)
        ax.plot([x.min() - 200, v], [q, q], color=plots.PALETTE["estimate"], lw=0.7,
                ls=":", zorder=2)
        ax.plot([v, v], [0, q], color=plots.PALETTE["estimate"], lw=0.7, ls=":", zorder=2)
        ax.annotate(f"{name} {v:.0f}g", xy=(v, q), xytext=(3, -8),
                    textcoords="offset points", fontsize=6.0,
                    color=plots.PALETTE["estimate"])
    ax.set_title(f"ECDF（n={x.size}、選ぶものは何もない）")
    ax.set_xlabel("体重 (g)")
    ax.set_ylabel("累積割合")
    ax.set_ylim(0, 1.02)

    ax = axes[1]
    for k, alpha in ((8, 0.75), (60, 0.55)):
        ax.hist(x, bins=k, density=True, alpha=alpha, lw=0,
                color=plots.PALETTE["data"] if k == 8 else plots.PALETTE["reject"])
    ax.annotate("8ビン", xy=(0.03, 0.92), xycoords="axes fraction", fontsize=6.2,
                color=plots.PALETTE["data"])
    ax.annotate("60ビン", xy=(0.03, 0.83), xycoords="axes fraction", fontsize=6.2,
                color=plots.PALETTE["reject"])
    ax.set_title("同じデータ、違うビン幅")
    ax.set_xlabel("体重 (g)")
    ax.set_ylabel("密度")
    plots.save(fig, "fig-2-5-ecdf.png")


if __name__ == "__main__":
    main()
