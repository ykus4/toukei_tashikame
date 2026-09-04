"""互いに無関係な100変数を並べると、必ず「強い相関」が見つかる。

100 変数を n=30 で観測すると、ペアは 100×99/2 = 4,950 通りある。真の相関はすべて 0 に
してあるので、見つかるものは全部まぐれである。それでも |r| が 0.5 を超えるペアは何十と
出るし、最大の |r| は 0.65 前後まで届く。散布図に描けば、誰が見ても「関係がある」形に
なる。

これは第9章の多重比較と同じ現象を、相関の側から見たものである。$\\alpha=0.05$ の検定を
4,950 回やれば、帰無仮説が全部正しくても平均 247.5 回は「有意」になる。探索的にヒート
マップを眺めて赤いマスを拾う作業は、この 247.5 回を拾う作業と区別がつかない。

    uv run python examples/ch11/spurious_correlations_100_variables.py
"""

import numpy as np
from scipy import stats

from toukei_tashikame import plots, sim

P, N, TRIALS, SEED = 100, 30, 1_000, 117
N_PAIRS = P * (P - 1) // 2
ALPHA = 0.05


def critical_r(n: int, alpha: float) -> float:
    """両側 alpha で有意になる |r| の下限。t 統計量を r に翻訳し直したもの。"""
    t = stats.t.ppf(1 - alpha / 2, df=n - 2)
    return float(t / np.sqrt(t**2 + n - 2))


R_CRIT = critical_r(N, ALPHA)
R_CRIT_BONF = critical_r(N, ALPHA / N_PAIRS)


def upper_triangle_r(rng: np.random.Generator) -> np.ndarray:
    """独立な P 変数 × n 観測を引き、全ペアの相関（上三角）を返す。"""
    x = rng.normal(size=(N, P))
    r = np.corrcoef(x, rowvar=False)
    return r[np.triu_indices(P, k=1)]


def one_trial(rng: np.random.Generator) -> tuple[float, float, float, float]:
    """1回ぶんの要約: 最大|r| / |r|>0.5 の数 / 有意な数 / Bonferroni 後の数。"""
    r = np.abs(upper_triangle_r(rng))
    return (float(r.max()), float((r > 0.5).sum()),
            float((r > R_CRIT).sum()), float((r > R_CRIT_BONF).sum()))


def main() -> None:
    plots.setup()
    print(f"--- 11-7 独立な {P} 変数、n={N}、{N_PAIRS:,} ペア（{TRIALS:,}回, seed={SEED}）---")
    print(f"  真の相関はすべて 0。α={ALPHA} で有意になる境界は |r| > {R_CRIT:.4f}")
    print(f"  Bonferroni（α/{N_PAIRS:,}）なら |r| > {R_CRIT_BONF:.4f}")

    out = sim.repeat(one_trial, trials=TRIALS, seed=SEED, progress=False)
    max_r, over_half, n_sig, n_bonf = out.T

    print(f"\n  最大 |r|            中央値 {np.median(max_r):.4f}   "
          f"平均 {max_r.mean():.4f}   範囲 [{max_r.min():.4f}, {max_r.max():.4f}]")
    print(f"  |r| > 0.5 のペア数  平均 {over_half.mean():.1f} 組   "
          f"中央値 {np.median(over_half):.0f} 組   最大 {over_half.max():.0f} 組")
    print(f"  p < {ALPHA} のペア数   平均 {n_sig.mean():.1f} 組"
          f"（期待値 {N_PAIRS * ALPHA:.1f} 組）")
    print(f"  Bonferroni 後       平均 {n_bonf.mean():.3f} 組   "
          f"1組以上出た試行の割合 {(n_bonf > 0).mean():.4f}（名目 {ALPHA}）")
    print(f"\n  |r|>0.5 が1組も出なかった試行 {int((over_half == 0).sum())} / {TRIALS:,}"
          f"   ← 最大 |r| の最小値でさえ {max_r.min():.4f} ある")
    print("  「探して見つけた」相関に p 値を付け直しても意味がない。"
          "探した回数のぶんだけ境界を動かすか、別のデータで確かめるかしかない")

    # 1回ぶんを取り出して、いちばん強く見えたペアがどう見えるかを確かめる。
    rng = np.random.default_rng(SEED)
    x = rng.normal(size=(N, P))
    r = np.corrcoef(x, rowvar=False)
    tri = np.triu_indices(P, k=1)
    k = int(np.argmax(np.abs(r[tri])))
    i, j = int(tri[0][k]), int(tri[1][k])
    best = float(r[i, j])
    print(f"\n--- この1回で最も強く見えたペア: 変数 {i} と 変数 {j} ---")
    print(f"  r = {best:+.4f}   p = {stats.pearsonr(x[:, i], x[:, j]).pvalue:.5f}   "
          f"r² = {best**2:.3f}")
    print("  真の相関は 0 である（そう作った）。それでも p 値は文句なしに小さい")

    # --- 図 ---
    fig, axes = plots.figure(1, 3, w=1.95, h=0.95)

    ax = axes[0]
    ax.hist(max_r, bins=40, color=plots.PALETTE["data"], alpha=0.6, lw=0)
    ax.axvline(0.5, color=plots.PALETTE["reject"], lw=1.1, ls="--", dashes=(4, 2.0), zorder=5)
    ax.annotate("|r| = 0.5", xy=(0.5, 0.9), xycoords=("data", "axes fraction"),
                xytext=(3, 0), textcoords="offset points", fontsize=6.0,
                color=plots.PALETTE["reject"])
    plots.mark_truth(ax, 0.0, "真値 ρ = 0")
    ax.set_xlim(0, max(0.55, max_r.max() * 1.02))
    ax.set_title(f"{N_PAIRS:,} ペア中の最大 |r|")
    ax.set_xlabel(f"最大 |r|（中央値 {np.median(max_r):.3f}）")
    ax.set_ylabel(f"{TRIALS:,} 回のうちの回数")

    ax = axes[1]
    ax.hist(n_sig, bins=40, color=plots.PALETTE["reject"], alpha=0.6, lw=0)
    plots.mark_truth(ax, N_PAIRS * ALPHA, f"期待値 {N_PAIRS * ALPHA:.1f}")
    ax.set_title("「有意」と出たペアの数")
    ax.set_xlabel(f"p < {ALPHA} のペア数")

    ax = axes[2]
    ax.scatter(x[:, i], x[:, j], s=8, color=plots.PALETTE["data"], lw=0, zorder=3)
    xs = np.array([x[:, i].min(), x[:, i].max()])
    b, a = np.polyfit(x[:, i], x[:, j], 1)
    ax.plot(xs, a + b * xs, color=plots.PALETTE["estimate"], lw=1.2, zorder=4)
    ax.set_title(f"最も強く見えたペア  r = {best:+.3f}（真値 0）")
    ax.set_xlabel(f"変数 {i}")
    ax.set_ylabel(f"変数 {j}")
    fig.tight_layout()
    plots.save(fig, "fig-11-7-spurious-max-correlation.png")


if __name__ == "__main__":
    main()
