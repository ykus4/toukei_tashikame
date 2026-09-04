"""68–95–99.7 を暗記するのではなく、1,000,000点を数えて確かめる。

$\\mathcal{N}(50, 10^2)$ から100万点を引き、$\\pm1\\sigma, \\pm2\\sigma, \\pm3\\sigma$
の中に入った点を数える。出てくるのは 0.68 / 0.95 / 0.997 で、μ や σ の値には依らない。
標準化 $z = (x-\\mu)/\\sigma$ をすれば、どんな正規分布も同じ1つの分布に重なるからで、
この「重なる」ことが平均±2SDという言い回しを支えている。

信頼区間の 1.96 も、外れ値の 3σ 則も、出どころはすべてこの表である。

    uv run python examples/ch04/normal_68_95_997_counted.py
"""

import numpy as np
from scipy import stats

from toukei_tashikame import plots

MU, SIGMA, N = 50.0, 10.0, 1_000_000
SEED = 14


def main() -> None:
    rng = np.random.default_rng(SEED)
    x = rng.normal(MU, SIGMA, size=N)

    print(f"--- N({MU:g}, {SIGMA:g}^2) から {N:,} 点 ---")
    print(f"  平均 {x.mean():.4f}（真値 {MU:g}） / SD {x.std(ddof=1):.4f}（真値 {SIGMA:g}）")

    print(f"\n{'範囲':<10}{'区間':>22}{'実測の割合':>13}{'理論':>10}{'外れる割合':>13}")
    for k in (1, 2, 3):
        lo, hi = MU - k * SIGMA, MU + k * SIGMA
        inside = float(((x >= lo) & (x <= hi)).mean())
        theory = float(stats.norm.cdf(k) - stats.norm.cdf(-k))
        print(f"{f'±{k}σ':<10}{f'[{lo:.0f}, {hi:.0f}]':>22}{inside:>13.4f}{theory:>10.4f}"
              f"{1 - inside:>13.4f}")
    print("  ±2σ の外は 20回に1回、±3σ の外は 370回に1回。"
          f"実測では {int((np.abs(x - MU) > 3 * SIGMA).sum()):,} 点が3σの外に出た")

    z = (x - MU) / SIGMA
    print("\n--- 標準化 z = (x−μ)/σ ---")
    print(f"  平均 {z.mean():+.4f}（理論 0） / SD {z.std(ddof=1):.4f}（理論 1）")
    print(f"  歪度 {stats.skew(z):+.4f}（理論 0） / 超過尖度 {stats.kurtosis(z):+.4f}（理論 0）")
    print("  μ と σ を変えても標準化すれば同じ1つの分布に重なる。"
          "だから表は1枚で足りる")

    print("\n--- よく使う分位点（標準正規）---")
    for p in (0.90, 0.95, 0.975, 0.99, 0.995):
        zq = stats.norm.ppf(p)
        emp = float(np.quantile(z, p))
        print(f"  ppf({p:.3f}) = {zq:.4f}   実測の分位点 {emp:.4f}")
    print("  信頼区間の 1.96 はここから来ている（第6章）")

    plots.setup()
    fig, ax = plots.figure()
    grid = np.linspace(MU - 4.2 * SIGMA, MU + 4.2 * SIGMA, 500)
    pdf = stats.norm.pdf(grid, MU, SIGMA)
    ax.hist(x, bins=120, density=True, color=plots.PALETTE["data"], alpha=0.45, lw=0)
    for k, alpha in ((1, 0.28), (2, 0.18), (3, 0.10)):
        band = (grid >= MU - k * SIGMA) & (grid <= MU + k * SIGMA)
        ax.fill_between(grid[band], pdf[band], color=plots.PALETTE["estimate"],
                        alpha=alpha, lw=0, zorder=1)
    ax.plot(grid, pdf, color=plots.PALETTE["truth"], lw=1.2, zorder=5)
    plots.mark_truth(ax, MU, "μ = 50")
    for k in (1, 2, 3):
        inside = float((np.abs(x - MU) <= k * SIGMA).mean())
        ax.annotate(f"±{k}σ: {inside:.4f}", xy=(MU + k * SIGMA, pdf.max() * (0.62 - 0.12 * k)),
                    fontsize=6.0, color=plots.PALETTE["estimate"], ha="left",
                    xytext=(2, 0), textcoords="offset points")
    ax.set_xlabel("x")
    ax.set_ylabel("密度")
    ax.set_title(f"N(50, 10^2) から {N:,} 点を数え上げる")
    plots.save(fig, "fig-4-6-68-95-997.png")


if __name__ == "__main__":
    main()
