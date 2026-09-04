"""r = 0 は「無関係」ではない — 完全に決まっているのに相関が消える3つの形。

$y=x^2$、$y=\\sin x$、そして円。どれも y は x から誤差なしに決まる（あるいは x と y が
1本の曲線の上に乗る）。関係は「強い」どころか完全である。それでも Pearson の r は 0 の
まわりに落ちる。r が測っているのは直線的な共変動だけで、上がってから下がる形は、上がる
ぶんと下がるぶんが打ち消し合って合計 0 になるからである。

Spearman なら救えるかというと、救えない。順位に直しても単調でないものは単調にならない。
一方、二次の項まで入れて当てはめれば $R^2$ は 1 に戻る。関係が無いのではなく、直線という
物差しがこの形を測れないだけだと分かる。

    uv run python examples/ch11/correlation_misses_curvature.py
"""

import numpy as np
from scipy import stats

from toukei_tashikame import plots

N, SEED = 500, 112


def relations(rng: np.random.Generator) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """3つの決定的な関係。誤差項は一切足していない。"""
    x1 = rng.uniform(-3.0, 3.0, size=N)
    x2 = rng.uniform(-np.pi / 2, 3 * np.pi / 2, size=N)   # 谷から谷まで、ちょうど1周期
    theta = rng.uniform(0.0, 2 * np.pi, size=N)
    return {
        "y = x²": (x1, x1**2),
        "y = sin x（1周期）": (x2, np.sin(x2)),
        "円 x²+y²=1": (np.cos(theta), np.sin(theta)),
    }


def curve_r2(x: np.ndarray, y: np.ndarray, deg: int) -> float:
    """次数 deg の多項式で当てはめたときの決定係数。deg=1 なら r² と一致する。"""
    resid = y - np.polyval(np.polyfit(x, y, deg), x)
    return 1.0 - resid.var() / y.var()


def main() -> None:
    plots.setup()
    rng = np.random.default_rng(SEED)
    data = relations(rng)

    print(f"--- 11-2 完全な関係、消える相関（各 n={N}, seed={SEED}）---")
    print("  関係                 Pearson r   p値      Spearman ρ   直線の R²   2次の R²")
    for name, (x, y) in data.items():
        pr = stats.pearsonr(x, y)
        sr = stats.spearmanr(x, y)
        print(f"  {name:<19} {pr.statistic: .4f}   {pr.pvalue:.3f}    {sr.statistic: .4f}"
              f"      {curve_r2(x, y, 1):.4f}      {curve_r2(x, y, 2):.4f}")

    x, y = data["y = x²"]
    print(f"\n  y = x² の左半分（x<0）だけなら r = {stats.pearsonr(x[x < 0], y[x < 0]).statistic:.4f}")
    print(f"  右半分（x>0）だけなら          r = {stats.pearsonr(x[x > 0], y[x > 0]).statistic:.4f}")
    print("  ← 打ち消し合っているだけで、どちら側も強い関係を持っている。"
          "全体の r=0 は「関係が無い」ではなく「向きが途中で変わる」の意味")
    print("  円は 2次の R² も 0 に近い。y は x の関数ですらない（1つの x に2つの y）ので、")
    print("  どんな回帰でも当てはまらない。それでも点は完全に円周の上にある")
    print("\n  だから相関係数を報告する前に散布図を描く。r は絵の要約であって、絵の代わりではない")

    # --- 図 ---
    fig, axes = plots.figure(1, 3, w=1.9, h=0.95)
    for ax, (name, (x, y)) in zip(axes, data.items(), strict=True):
        ax.scatter(x, y, s=4, color=plots.PALETTE["data"], lw=0, alpha=0.7, zorder=3)
        b, a = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 100)
        ax.plot(xs, a + b * xs, color=plots.PALETTE["estimate"], lw=1.2, zorder=4)
        r = stats.pearsonr(x, y).statistic
        rho = stats.spearmanr(x, y).statistic
        ax.set_title(f"{name}   r = {r:.4f}")
        ax.annotate(f"Spearman ρ = {rho:.4f}\n最小二乗の直線（青）", xy=(0.5, 0.04),
                    xycoords="axes fraction", ha="center", fontsize=6.0,
                    color=plots.PALETTE["estimate"])
        ax.set_xlabel("x")
    axes[0].set_ylabel("y")
    fig.tight_layout()
    plots.save(fig, "fig-11-2-quadratic-zero-correlation.png")


if __name__ == "__main__":
    main()
