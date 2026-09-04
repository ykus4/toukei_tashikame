"""nを増やせば、どんなに小さい差でも「有意」になる。

真の差を 0.01（標準偏差の1%）に固定したまま n だけを増やす。差は最初から最後まで
まったく同じで、実務的にはどう見ても無意味な大きさである。それでも n=10^6 では
ほぼ毎回 p < 0.05 になる。

有意性は「差の大きさ」ではなく「差の大きさ × サンプルサイズ」を測っている。だから
巨大なログデータでの検定は、ほぼ必ず有意になる。報告すべきは p値ではなく効果量と
信頼区間である、という話がここから出る。

n=10^6 の標本を1,000回引くと数十秒かかるので、正規母集団の十分統計量
（標本平均と標本分散）の分布から直接引く。データを並べてから平均を取るのと
数学的に同じで、n=10^2 と 10^3 では実際にデータを作った結果と突き合わせてある。

    uv run python examples/ch07/pvalue_shrinks_with_n.py
"""

import numpy as np
from scipy import stats

from toukei_tashikame import plots, sim, testing

DELTA = 0.01      # 真の差。σ=1 なので Cohen の d = 0.010 で固定
ALPHA = 0.05
TRIALS = 1_000
NS = (10**2, 10**3, 10**4, 10**5, 10**6)


def sufficient_stats_pvalues(n: int, rng) -> tuple[np.ndarray, np.ndarray]:
    """n個の標本を並べずに、t検定のp値を TRIALS 個作る。

    正規母集団では x̄ ~ N(μ, σ²/n)、(n-1)s²/σ² ~ χ²(n-1) で、この2つは独立。
    t = x̄ / (s/√n) はデータを作ってから計算しても同じ分布になる。
    """
    xbar = rng.normal(DELTA, 1.0 / np.sqrt(n), size=TRIALS)
    s2 = rng.chisquare(n - 1, size=TRIALS) / (n - 1)
    t = xbar / np.sqrt(s2 / n)
    p = 2 * stats.t.sf(np.abs(t), df=n - 1)
    return p, xbar / np.sqrt(s2)   # p値と、その標本から推定した効果量 d


def direct_pvalues(n: int, seed: int) -> np.ndarray:
    """突き合わせ用。実際に n 個のデータを作って t 検定にかける。"""

    def one(rng):
        x = rng.normal(DELTA, 1.0, size=n)
        return testing.t_1samp(x, 0.0).pvalue

    return sim.repeat(one, trials=TRIALS, seed=seed, progress=False)


def main() -> None:
    plots.setup()

    rng = np.random.default_rng(79)
    rates, medians, ds = [], [], []

    print(f"--- 真の差を {DELTA} に固定して n だけを増やす"
          f"（各 {TRIALS:,} 回, α={ALPHA}, seed=79）---")
    print("          n     棄却率     p値の中央値     推定した効果量 d の平均")
    for n in NS:
        p, d_hat = sufficient_stats_pvalues(n, rng)
        rate = float((p < ALPHA).mean())
        rates.append(rate)
        medians.append(float(np.median(p)))
        ds.append(float(d_hat.mean()))
        print(f"  {n:>9,}   {rate:.4f}      {np.median(p):.3e}          {d_hat.mean():+.4f}")

    print(f"\n  効果量は最後まで d ≈ {DELTA:.3f} のまま。動いたのは n だけである")
    print(f"  n={NS[0]:,} では棄却率 {rates[0]:.4f}（α とほぼ同じ = ほぼ見つけられない）、")
    print(f"  n={NS[-1]:,} では {rates[-1]:.4f}。同じ差が「有意でない」から"
          "「ほぼ確実に有意」に変わる")
    print("  変わったのは世界ではなく、こちらの目の細かさ。"
          "p値は差の大きさを測っていない")

    # --- 十分統計量からの近道が、実データと一致することの確認 ---
    print("\n--- 突き合わせ: 実際にデータを作った場合の棄却率 ---")
    for n, seed in ((NS[0], 790), (NS[1], 791)):
        direct = direct_pvalues(n, seed)
        i = NS.index(n)
        print(f"  n={n:>6,}   十分統計量 {rates[i]:.4f}   実データ "
              f"{float((direct < ALPHA).mean()):.4f}"
              f"   （{TRIALS:,}回のシミュレーション誤差は ±0.014 程度）")

    # --- 図 ---
    fig, axes = plots.figure(1, 2)
    ax = axes[0]
    ax.plot(NS, rates, "o-", ms=3.5, color=plots.PALETTE["estimate"], zorder=4)
    ax.axhline(ALPHA, color=plots.PALETTE["reject"], lw=0.9, ls="--", dashes=(4, 2.2))
    ax.annotate("α = 0.05", xy=(NS[0], ALPHA), xytext=(2, 4), textcoords="offset points",
                fontsize=6.0, color=plots.PALETTE["reject"])
    ax.set_xscale("log")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("サンプルサイズ n")
    ax.set_ylabel(f"p < {ALPHA} になった割合")
    ax.set_title(f"真の差は {DELTA} のまま動かない")

    ax = axes[1]
    ax.plot(NS, medians, "o-", ms=3.5, color=plots.PALETTE["estimate"], zorder=4)
    ax.axhline(ALPHA, color=plots.PALETTE["reject"], lw=0.9, ls="--", dashes=(4, 2.2))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("サンプルサイズ n")
    ax.set_ylabel("p値の中央値")
    ax.set_title("p値は n とともにいくらでも小さくなる")
    ax.annotate("α = 0.05", xy=(NS[0], ALPHA), xytext=(2, -9),
                textcoords="offset points", fontsize=6.0, color=plots.PALETTE["reject"])
    ax.annotate(f"真の効果量は d = {DELTA:.3f} で不変。\n小さくなるのは p値だけ",
                xy=(0.06, 0.40), xycoords="axes fraction", fontsize=6.0,
                color=plots.PALETTE["truth"])
    fig.tight_layout()
    plots.save(fig, "fig-7-9-pvalue-vs-n.png")


if __name__ == "__main__":
    main()
