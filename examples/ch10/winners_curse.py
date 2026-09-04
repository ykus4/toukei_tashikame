"""勝者の呪い — 検出力が低い設計で「有意になった」効果量は、必ず大きすぎる。

真の効果は 0.20 で固定し、検出力 0.2 / 0.5 / 0.8 の3つの設計で 10,000 回ずつ実験する。
全試行の効果量の平均は、どの設計でも 0.20 に当たる。推定はバイアスしていない。

ところが**有意になった試行だけ**を拾うと平均は跳ね上がる。有意になるには t が閾値を
超える必要があり、検出力が低い設計では「たまたま大きく出た」ときにしかそこへ届かない。
発表されるのは有意なものだけなので、世に出る効果量は系統的に過大になる。p ハッキングを
一度もしていなくてもこうなる——足切りをした標本の性質そのものである。

さらに悪いことに、有意なのに符号が逆（第S種の誤り）まで起きる。

    uv run python examples/ch10/winners_curse.py
"""

import numpy as np
from scipy import stats

from toukei_tashikame import plots, power

D_TRUE, ALPHA, TRIALS, SEED = 0.20, 0.05, 10_000, 105
DESIGNS = (0.2, 0.5, 0.8)      # 目標の検出力


def trial_batch(n: int, d_true: float, trials: int, rng):
    """(trials, n) を2枚引いて Welch の t 検定をまとめてかけ、(p値, 観測効果量) を返す。"""
    a = rng.normal(0.0, 1.0, size=(trials, n))
    b = rng.normal(d_true, 1.0, size=(trials, n))
    ma, mb = a.mean(axis=1), b.mean(axis=1)
    va, vb = a.var(axis=1, ddof=1), b.var(axis=1, ddof=1)
    se = np.sqrt(va / n + vb / n)
    t = (mb - ma) / se
    df = se**4 / ((va / n) ** 2 / (n - 1) + (vb / n) ** 2 / (n - 1))
    p = 2 * stats.t.sf(np.abs(t), df)
    d_hat = (mb - ma) / np.sqrt((va + vb) / 2)     # Cohen's d（プールした標準偏差）
    return p, d_hat


def main() -> None:
    plots.setup()
    rng = np.random.default_rng(SEED)

    print(f"--- 真の効果 d={D_TRUE} を、検出力の違う3つの設計で {TRIALS:,} 回ずつ ---")
    print("  設計検出力   n/群   実測検出力   全試行の平均   有意な試行の平均   倍率   符号が逆")
    results = {}
    for target in DESIGNS:
        n = power.n_for_power(D_TRUE, power=target, alpha=ALPHA)
        p, d_hat = trial_batch(n, D_TRUE, TRIALS, rng)
        sig = p < ALPHA
        d_sig = d_hat[sig]
        wrong_sign = float((d_sig < 0).mean())
        results[target] = (n, d_hat, sig)
        print(f"     {target:.1f}     {n:5d}     {sig.mean():.4f}"
              f"        {d_hat.mean():.4f}          {d_sig.mean():.4f}"
              f"       {d_sig.mean() / D_TRUE:.2f}倍   {wrong_sign:.4f}")

    print("\n  推定そのものは正しい（全試行の平均は真値 0.20 に当たっている）。")
    print("  歪むのは「有意なものだけを見る」という選び方のほうである。")
    print("  検出力 0.2 の設計で有意になった論文の効果量は、真値の2倍以上と読むべき。")

    # 道具（power.winners_curse）と突き合わせる。手書きの t 検定が同じ答えを出すか。
    n80 = results[0.8][0]
    lib = power.winners_curse(n80, D_TRUE, alpha=ALPHA, trials=2_000, seed=SEED)
    p2, d2 = trial_batch(n80, D_TRUE, 2_000, np.random.default_rng(SEED + 1))
    print(f"\n--- 照合（n={n80}, 2,000回）---")
    print(f"  power.winners_curse   有意試行の平均 {lib['d_significant']:.4f}"
          f"  検出力 {lib['power']:.4f}")
    print(f"  このスクリプトの手書き 有意試行の平均 {d2[p2 < ALPHA].mean():.4f}"
          f"  検出力 {(p2 < ALPHA).mean():.4f}   ← 乱数が違うぶんだけずれる")

    # --- 図 ---
    fig, axes = plots.figure(1, 3, w=1.9, h=0.95, sharey=True)
    bins = np.linspace(-0.5, 0.9, 60)
    for ax, target in zip(axes, DESIGNS, strict=True):
        n, d_hat, sig = results[target]
        ax.hist(d_hat, bins=bins, color=plots.PALETTE["data"], alpha=0.45, lw=0)
        ax.hist(d_hat[sig], bins=bins, color=plots.PALETTE["reject"], alpha=0.75, lw=0)
        plots.mark_truth(ax, D_TRUE, f"真値 {D_TRUE}")
        ax.axvline(d_hat[sig].mean(), color=plots.PALETTE["estimate"], lw=1.1,
                   ls="--", dashes=(4, 2.0), zorder=6)
        ax.annotate(f"有意な試行の平均 {d_hat[sig].mean():.3f}", xy=(0.97, 0.72),
                    xycoords="axes fraction", ha="right", fontsize=6.0,
                    color=plots.PALETTE["estimate"])
        ax.set_title(f"検出力 {target:.1f}（n={n}/群）")
        ax.set_xlabel("観測された効果量 $\\hat{d}$")
    axes[0].set_ylabel(f"{TRIALS:,} 回のうちの回数")
    fig.tight_layout()
    plots.save(fig, "fig-10-5-winners-curse.png")


if __name__ == "__main__":
    main()
