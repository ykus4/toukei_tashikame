"""事後検出力はp値の言い換えでしかない — 散布図が1本の曲線になる。

検定のあとで「観測された効果量」を入れて計算した検出力を事後検出力（observed power）
という。有意にならなかった実験の報告に「事後検出力は 0.31 だったので検出力不足です」と
書かれることがあるが、これは何の追加情報でもない。

同じ n の実験では、観測効果量とp値は1対1に対応する。だから事後検出力もp値の関数に
なり、10,000回の実験を散布図にすると点は**1本の曲線に乗る**。Spearman 相関は -1。
とくに p=0.05 ちょうどのとき事後検出力はいつでもほぼ 0.5 になる。

検出力は設計のための量であり、n と**知りたい効果量**から前もって決めるもの。データを
見たあとに計算しても、p値を別の目盛りで読み直しただけになる。

    uv run python examples/ch10/post_hoc_power_is_a_function_of_p.py
"""

import numpy as np
from scipy import stats

from toukei_tashikame import plots, power

N, D_TRUE, ALPHA, TRIALS, SEED = 30, 0.3, 0.05, 10_000, 106


def main() -> None:
    plots.setup()
    rng = np.random.default_rng(SEED)
    df = 2 * (N - 1)
    crit = stats.t.ppf(1 - ALPHA / 2, df)

    # --- 10,000回の実験。p値と観測効果量を出す（等分散を仮定した t 検定）---
    a = rng.normal(0.0, 1.0, size=(TRIALS, N))
    b = rng.normal(D_TRUE, 1.0, size=(TRIALS, N))
    sp = np.sqrt((a.var(axis=1, ddof=1) + b.var(axis=1, ddof=1)) / 2)
    d_hat = (b.mean(axis=1) - a.mean(axis=1)) / sp
    t = d_hat * np.sqrt(N / 2)
    p = 2 * stats.t.sf(np.abs(t), df)

    # --- 事後検出力。観測効果量をそのまま「真の効果」だと思って検出力の式に入れる ---
    ncp = np.abs(d_hat) * np.sqrt(N / 2)
    post_hoc = stats.nct.sf(crit, df, ncp) + stats.nct.cdf(-crit, df, ncp)

    true_power = power.power_ttest(N, D_TRUE, alpha=ALPHA)
    rho = stats.spearmanr(p, post_hoc).statistic

    print(f"--- n={N}/群、真の効果 d={D_TRUE} の実験を {TRIALS:,} 回 ---")
    print(f"  設計時の（本当の）検出力      {true_power:.4f}")
    print(f"  事後検出力の平均              {post_hoc.mean():.4f}")
    print(f"  p値と事後検出力の Spearman 相関 {rho:.4f}   ← 1本の曲線に乗っている")

    print("\n  p値ごとの事後検出力（同じp値なら、いつでも同じ事後検出力）")
    for target in (0.001, 0.01, 0.05, 0.10, 0.30, 0.80):
        tt = stats.t.isf(target / 2, df)
        ph = stats.nct.sf(crit, df, tt) + stats.nct.cdf(-crit, df, tt)
        print(f"    p = {target:<5}  →  事後検出力 {ph:.4f}")
    near = np.abs(p - 0.05) < 0.002
    print(f"  実測でも p≈0.05 だった {near.sum()} 回の事後検出力は "
          f"{post_hoc[near].min():.4f}〜{post_hoc[near].max():.4f}")

    print("\n  有意にならなかった実験の事後検出力は、0.5 を超えられない:")
    print(f"    p ≥ 0.05 の試行 {int((p >= ALPHA).sum()):,} 回の事後検出力の最大値 "
          f"{post_hoc[p >= ALPHA].max():.4f}")
    print("  「検出力が低かった」という報告は「有意でなかった」と同じことを言っている。")

    # --- 図 ---
    fig, ax = plots.figure(w=1.15, h=1.05)
    idx = rng.choice(TRIALS, size=2000, replace=False)     # 描画は間引く
    ax.scatter(p[idx], post_hoc[idx], s=3, lw=0, alpha=0.5,
               color=plots.PALETTE["data"], zorder=3)
    ax.axvline(ALPHA, color=plots.PALETTE["reject"], lw=0.9, ls="--", dashes=(4, 2.2),
               zorder=4)
    ax.axhline(0.5, color=plots.PALETTE["reject"], lw=0.9, ls="--", dashes=(4, 2.2),
               zorder=4)
    ax.annotate("p = 0.05 なら事後検出力はほぼ 0.5", xy=(0.02, 0.54),
                xycoords="axes fraction", fontsize=6.0,
                color=plots.PALETTE["reject"])
    plots.mark_truth(ax, true_power, f"設計時の検出力 {true_power:.3f}", axis="y")
    ax.set_xscale("log")
    ax.set_xlabel("p値（対数目盛）")
    ax.set_ylabel("事後検出力")
    ax.set_ylim(0, 1.02)
    ax.set_title(f"{TRIALS:,} 回の実験（n={N}/群、d={D_TRUE}）から2,000点を表示")
    fig.tight_layout()
    plots.save(fig, "fig-10-6-post-hoc-power-vs-p.png")


if __name__ == "__main__":
    main()
