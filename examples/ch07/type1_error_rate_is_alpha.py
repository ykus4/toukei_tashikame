"""αは「約束した誤警報の率」であって、それ以上でも以下でもない。

差がまったく無い2群を10,000回作って検定すると、α=0.05 なら約500回は「差がある」と
言ってしまう。これは検定の失敗ではなく仕様である。αは、実装者が自分で選んで受け入れた
誤警報の率にすぎない。

同じ10,000個のp値に、0.01 / 0.05 / 0.10 の3つのものさしを当てる。データは1つで、
変えているのは切る位置だけ。棄却率が切った位置そのものになることを見る。

    uv run python examples/ch07/type1_error_rate_is_alpha.py
"""

import numpy as np

from toukei_tashikame import plots, sim, testing

N = 25
TRIALS = 10_000
ALPHAS = (0.01, 0.05, 0.10)


def null_pvalue(rng) -> float:
    """真の差がゼロの2群（各 n=25）を作って Welch のt検定にかける。"""
    a = rng.normal(0.0, 1.0, size=N)
    b = rng.normal(0.0, 1.0, size=N)
    return testing.t_ind(a, b).pvalue


def main() -> None:
    plots.setup()

    p = sim.repeat(null_pvalue, trials=TRIALS, seed=705, progress=False)

    print(f"--- 真の差ゼロの2群（各 n={N}）を {TRIALS:,} 回検定（seed=705）---")
    print("   α      棄却した回数    実測の第一種の誤り     95%区間")
    rates = []
    for alpha in ALPHAS:
        k = int((p < alpha).sum())
        rate = k / TRIALS
        se = np.sqrt(rate * (1 - rate) / TRIALS)
        rates.append((rate, se))
        print(f"  {alpha:.2f}     {k:>5,} / {TRIALS:,}       {rate:.4f}"
              f"        [{rate - 1.96 * se:.4f}, {rate + 1.96 * se:.4f}]")

    print("\n  3行とも同じ10,000個のp値を使っている。変えたのは切る位置だけで、")
    print("  棄却率は切った位置そのものになる。αは検出したい何かではなく、")
    print("  「間違って騒ぐ回数をこれ以下に抑える」という自分への制約である")
    print(f"\n  10,000回のシミュレーション誤差は α=0.05 付近で ±{1.96 * rates[1][1]:.4f}。")
    print("  この幅より小さいずれは、数え直すだけで動く")

    # --- 図: 名目のαと実測の棄却率 ---
    fig, axes = plots.figure(1, 2)
    ax = axes[0]
    xs = np.linspace(0, 0.12, 200)
    ax.plot(xs, xs, color=plots.PALETTE["truth"], lw=1.1, zorder=4)
    ax.annotate("実測 = α\nこの線に乗れば設計どおり", xy=(0.004, 0.119), va="top",
                fontsize=6.0, color=plots.PALETTE["truth"])
    ax.errorbar(ALPHAS, [r for r, _ in rates], yerr=[1.96 * s for _, s in rates],
                fmt="o", ms=3.5, lw=1.0, capsize=2.0,
                color=plots.PALETTE["estimate"], zorder=5)
    for alpha, (rate, _) in zip(ALPHAS, rates, strict=True):
        ax.annotate(f"{rate:.4f}", xy=(alpha, rate), xytext=(4, -6),
                    textcoords="offset points", fontsize=6.0,
                    color=plots.PALETTE["estimate"])
    ax.set_xlabel("名目の α")
    ax.set_ylabel("実測の第一種の誤り")
    ax.set_title("約束した率が、そのまま出る")

    ax = axes[1]
    xs = np.sort(p)[: int(TRIALS * 0.15)]
    ax.plot(xs, np.arange(1, xs.size + 1) / TRIALS, color=plots.PALETTE["estimate"], lw=1.2)
    ax.plot([0, 0.15], [0, 0.15], color=plots.PALETTE["truth"], lw=1.0, ls="--",
            dashes=(4, 2.0))
    for alpha in ALPHAS:
        ax.axvline(alpha, color=plots.PALETTE["reject"], lw=0.8, ls="--", dashes=(3, 2.0))
    ax.set_xlim(0, 0.15)
    ax.set_xlabel("p値（下側だけ拡大）")
    ax.set_ylabel("そこまでに棄却した割合")
    ax.set_title("3本の縦線が3つの α")
    fig.tight_layout()
    plots.save(fig, "fig-7-5-type1-error-rate.png")


if __name__ == "__main__":
    main()
