"""検出力と第二種の誤りは、1枚の絵の中の2つの面積である。

帰無分布と対立分布は重なっている。棄却域の線を1本引くと、その線は帰無分布の裾
（= α、間違って騒ぐ確率）と、対立分布の内側（= β、本当の差を見逃す確率）を同時に
切る。片方を小さくすればもう片方が大きくなる。これがトレードオフの正体で、
「検出力を上げたい」は本来「重なりを減らしたい」＝ n を増やすか差が大きいか、である。

真の差 δ=0.5（各群 n=30）の世界を10,000回作り、棄却域に落ちた割合を数える。

    uv run python examples/ch07/null_vs_alternative_overlap.py
"""

import numpy as np
from scipy import stats

from toukei_tashikame import plots, power, sim, testing

N = 30
DELTA = 0.5      # 真の差（σ=1 なので Cohen の d もこの値）
ALPHA = 0.05
TRIALS = 10_000
NULL_TRIALS = 2_000   # 帰無側は添え物。α は 7-5 で 10,000 回数えてある


def alt_pvalue(rng) -> float:
    """真の差が δ の2群を引いて、Welch のt検定のp値を返す。"""
    a = rng.normal(0.0, 1.0, size=N)
    b = rng.normal(DELTA, 1.0, size=N)
    return testing.t_ind(a, b).pvalue


def null_pvalue(rng) -> float:
    """比較のための、差ゼロの世界。手続きは1文字も変わらない。"""
    a = rng.normal(0.0, 1.0, size=N)
    b = rng.normal(0.0, 1.0, size=N)
    return testing.t_ind(a, b).pvalue


def main() -> None:
    plots.setup()

    alt = sim.rejection_rate(alt_pvalue, alpha=ALPHA, trials=TRIALS, seed=76, progress=False)
    null = sim.rejection_rate(null_pvalue, alpha=ALPHA, trials=NULL_TRIALS, seed=76,
                              progress=False)
    theory = power.power_ttest(N, DELTA, ALPHA, kind="two-sample")

    print(f"--- 真の差 δ={DELTA}、各群 n={N}、α={ALPHA} を {TRIALS:,} 回 ---")
    print(f"  対立が真の世界で棄却した割合  {alt.rate:.4f} ± {1.96 * alt.se:.4f}  ← 検出力")
    print(f"  見逃した割合                  {1 - alt.rate:.4f}"
          f"              ← 第二種の誤り β")
    print(f"  非心t分布から解いた検出力     {theory:.4f}")
    print(f"\n  帰無が真の世界で棄却した割合  {null.rate:.4f} ± {1.96 * null.se:.4f}"
          f"  ← α（{NULL_TRIALS:,}回）")
    print("  数えるコードは2つとも同じで、違うのはデータの作り方だけ")

    missed = round((1 - alt.rate) * TRIALS)
    print(f"\n  本当に差がある世界を {TRIALS:,} 回まわして、{missed:,} 回は"
          "「有意差なし」と結論している。")
    print("  「有意でなかった」は「差が無い」ではない。この設計では、差があっても"
          "半分ちかくを取り逃がす")

    # --- 図: 2つの分布と、1本の線が切る2つの面積 ---
    crit = stats.t.ppf(1 - ALPHA / 2, df=2 * (N - 1))
    ncp = DELTA * np.sqrt(N / 2)
    grid = np.linspace(-4.5, 7.0, 600)
    null_y = stats.t.pdf(grid, df=2 * (N - 1))
    alt_y = stats.nct.pdf(grid, df=2 * (N - 1), nc=ncp)

    fig, ax = plots.figure(h=1.05)
    plots.null_vs_alt(ax, grid, null_y, grid, alt_y, crit=crit, tail="upper")
    keep = grid < crit
    ax.fill_between(grid[keep], alt_y[keep], color=plots.PALETTE["alt"], alpha=0.22, lw=0,
                    zorder=1)
    ax.fill_between(grid[~keep], alt_y[~keep], color=plots.PALETTE["alt"], alpha=0.40, lw=0,
                    zorder=1)
    ax.annotate("帰無分布 H₀: 差なし", xy=(-2.6, 0.30), fontsize=6.2,
                color=plots.PALETTE["ink2"])
    ax.annotate(f"対立分布 H₁: δ={DELTA}", xy=(3.1, 0.30), fontsize=6.2,
                color=plots.PALETTE["alt"])
    ax.annotate(f"β = {1 - alt.rate:.4f}\n（見逃し）", xy=(0.9, 0.10), fontsize=6.0,
                color=plots.PALETTE["alt"], ha="center")
    ax.annotate(f"検出力 = {alt.rate:.4f}\n（対立分布の、線より右）", xy=(3.4, 0.16),
                fontsize=6.0, color=plots.PALETTE["alt"])
    ax.annotate(f"α = {ALPHA}\n（帰無分布の裾）", xy=(2.7, 0.010), xytext=(4.6, 0.055),
                fontsize=6.0, color=plots.PALETTE["reject"], ha="center",
                arrowprops={"arrowstyle": "-", "color": plots.PALETTE["reject"], "lw": 0.7})
    ax.set_xlabel("t 統計量")
    ax.set_ylabel("密度")
    ax.set_title("1本の棄却線が、αとβを同時に決める")
    plots.save(fig, "fig-7-6-null-alt-overlap.png")


if __name__ == "__main__":
    main()
