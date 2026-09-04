"""MSE = バイアス² + バリアンス を数値で確かめ、「偏らせたほうが得」な場合を見る。

縮小推定量 c·x̄ の c を掃く。c=1 が不偏で、c を 1 から下げるとバイアスが増えるかわりに
バリアンスが減る。合計である MSE がどちらに転ぶかは、真値 μ と σ/√n の大小で決まる。
不偏性は目的ではなく、選択肢の1つでしかない。

    uv run python examples/ch06/bias_variance_decomposition_mse.py
"""

import numpy as np

from toukei_tashikame import estimate, plots, sim

SIGMA, N = 10.0, 20
TRIALS, SEED = 10_000, 25
# 信号（μ）に対して雑音（σ/√n）が小さい場合と大きい場合を並べる。
SETTINGS = (
    ("μ=50（雑音が小さい）", 50.0, np.round(np.arange(0.90, 1.0501, 0.01), 3)),
    ("μ=5（雑音が大きい）", 5.0, np.round(np.arange(0.50, 1.0501, 0.05), 3)),
)


def sweep_c(mu: float, cs: np.ndarray) -> np.ndarray:
    """各 c について ``(bias2, variance, mse)`` を返す。標本は c 間で共通に使う。"""
    xbar = sim.repeat(lambda rng: float(rng.normal(mu, SIGMA, size=N).mean()),
                      trials=TRIALS, seed=SEED, progress=False)
    return np.array([estimate.mse_decomposition(c * xbar, truth=mu) for c in cs])


def main() -> None:
    plots.setup()
    fig, axes = plots.figure(1, 2, w=2.0)
    results = []

    for ax, (label, mu, cs) in zip(axes, SETTINGS, strict=True):
        table = sweep_c(mu, cs)
        bias2, var, mse = table[:, 0], table[:, 1], table[:, 2]
        best = int(np.argmin(mse))
        c_star = mu**2 / (mu**2 + SIGMA**2 / N)   # 理論上の最適な c
        results.append((label, mu, cs, table, best, c_star))

        print(f"--- {label}: c·x̄ の c を {cs[0]:.2f}〜{cs[-1]:.2f} で掃く（各 {TRIALS:,} 回）---")
        print(f"  {'c':>6}{'バイアス²':>12}{'バリアンス':>12}{'MSE':>10}{'恒等式の残差':>14}")
        for i, c in enumerate(cs):
            mark = "  ← 最小" if i == best else ""
            print(f"  {c:>6.2f}{bias2[i]:>12.4f}{var[i]:>12.4f}{mse[i]:>10.4f}"
                  f"{abs(mse[i] - bias2[i] - var[i]):>14.7f}{mark}")
        print(f"  恒等式 MSE = バイアス² + バリアンス の残差は最大 "
              f"{np.abs(mse - bias2 - var).max():.2e}（浮動小数の丸めだけ）")
        print(f"  MSE が最小なのは c={cs[best]:.2f}（{mse[best]:.4f}）。"
              f"不偏な c=1.00 は {mse[np.argmin(np.abs(cs - 1.0))]:.4f}")
        print(f"  理論上の最適 c* = μ²/(μ²+σ²/n) = {c_star:.4f}\n")

        ax.plot(cs, mse, color=plots.PALETTE["estimate"], lw=1.4)
        ax.plot(cs, var, color=plots.PALETTE["data"], lw=1.0)
        ax.plot(cs, bias2, color=plots.PALETTE["ink2"], lw=1.0, ls="--", dashes=(4, 2.0))
        ax.axvline(1.0, color=plots.PALETTE["ink2"], lw=0.7, alpha=0.6)
        ax.annotate("c=1（不偏）", xy=(1.0, 0.98), xycoords=("data", "axes fraction"),
                    ha="right", va="top", fontsize=6.0, color=plots.PALETTE["ink2"])
        ax.annotate("MSE", xy=(cs[0], mse[0]), fontsize=6.0, color=plots.PALETTE["estimate"])
        ax.annotate("バリアンス", xy=(cs[0], var[0]), xytext=(1, -7), textcoords="offset points",
                    fontsize=6.0, color=plots.PALETTE["data"])
        ax.annotate("バイアス²", xy=(cs[0], bias2[0]), xytext=(1, 3), textcoords="offset points",
                    fontsize=6.0, color=plots.PALETTE["ink2"])
        ax.set_xlabel("縮小の係数 c")
        ax.set_ylabel("μ の推定の誤差（2乗）")
        ax.set_title(label)

    winner = results[1]
    gain = 1 - winner[3][winner[4], 2] / winner[3][np.argmin(np.abs(winner[2] - 1.0)), 2]
    print(f"--- まとめ ---\n  μ=5 では c={winner[2][winner[4]]:.2f} まで縮めると MSE が "
          f"{100 * gain:.1f}% 小さい。わざと偏らせたほうが当たる")
    print("  μ=50 では最適 c* が 1 のすぐ手前にあり、縮めても得はほとんどない")
    plots.save(fig, "fig-6-3-bias-variance.png")


if __name__ == "__main__":
    main()
