"""95%信頼区間の「95%」を数え、最初の100本を縞模様で描く。

μ=50 の母集団から n=20 を引いて t 区間を作る、を 10,000 回。真値を包んだ本数を数えると
0.95 の付近に落ちる。動いているのは区間のほうで、μ は縦の1本の直線として一度も動かない。
「95%の確率で μ がこの区間にある」ではなく「この作り方を繰り返すと95%が当たる」である。

    uv run python examples/ch06/confidence_interval_100_stripes.py
"""

import numpy as np

from toukei_tashikame import estimate, plots, sim

MU, SIGMA, N = 50.0, 10.0, 20
TRIALS, SEED, CONF = 10_000, 27, 0.95
N_SHOW = 100


def one_trial(rng) -> tuple[float, float]:
    """標本を1つ引いて t 信頼区間を返す。σ は既知としない（現場と同じ条件）。"""
    return estimate.ci_mean_t(rng.normal(MU, SIGMA, size=N), conf=CONF)


def main() -> None:
    plots.setup()
    res = sim.coverage(one_trial, truth=MU, trials=TRIALS, seed=SEED, progress=False)
    iv = res.intervals
    widths = iv[:, 1] - iv[:, 0]

    print(f"--- t 区間（n={N}, 名目 {CONF:.0%}）を {TRIALS:,} 本 ---")
    print(f"  被覆       {res.rate:.4f} ± {1.96 * res.se:.4f}（数え直しの誤差込み）")
    print(f"  外した本数 {TRIALS - int(res.covered.sum()):,} / {TRIALS:,}")
    below = int(np.sum(iv[:, 1] < MU))
    above = int(np.sum(iv[:, 0] > MU))
    print(f"  内訳       下に外した {below} 本 / 上に外した {above} 本（正規なので左右ほぼ対称）")
    print(f"  区間の幅   平均 {widths.mean():.2f} / 最小 {widths.min():.2f} / 最大 {widths.max():.2f}"
          "   ← 幅も標本ごとに動く")

    missed = np.flatnonzero(~res.covered[:N_SHOW]) + 1
    print(f"\n--- 最初の {N_SHOW} 本 ---")
    print(f"  外した区間 {missed.size} 本（{'、'.join(f'{i} 本目' for i in missed)}）")
    for i in missed:
        lo, hi = iv[i - 1]
        side = "下に" if hi < MU else "上に"
        print(f"    {i:>3} 本目  [{lo:.2f}, {hi:.2f}]   {side}外した")
    print(f"  100 本なら外れは平均 5 本。{missed.size} 本はその揺らぎの中"
          f"（±{1.96 * np.sqrt(0.05 * 0.95 / N_SHOW) * N_SHOW:.1f} 本）")

    fig, ax = plots.figure(h=1.5)
    n_missed = plots.coverage_stripes(ax, iv, MU, n_show=N_SHOW)
    ax.set_xlabel("μ の 95% 信頼区間")
    ax.set_title(f"{N_SHOW} 本中 {n_missed} 本が真値を外す（全 {TRIALS:,} 本の被覆 {res.rate:.4f}）")
    plots.save(fig, "fig-6-5-ci-stripes.png")


if __name__ == "__main__":
    main()
