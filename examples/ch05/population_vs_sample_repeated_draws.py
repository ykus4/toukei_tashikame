"""母集団と標本 — 「標本平均」という数字そのものがばらつくことを数え上げる。

母集団（サイズ1,000,000、μ=50, σ=10）を自分で作ってしまえば、真値は隠れていない。
そこから n=25 の標本を引き、標本平均を計算する。この操作を10,000回くり返すと、
標本平均は毎回ちがう値になる。つまり**標本平均は定数ではなく確率変数**である。

母平均 μ は一度も動かない。動いているのは標本のほうで、これが第5章の全体を貫く見方に
なる。10,000個の標本平均を並べたもの——それが標本分布である。

    uv run python examples/ch05/population_vs_sample_repeated_draws.py
"""

import numpy as np

from toukei_tashikame import plots, sim

MU, SIGMA = 50.0, 10.0
POP_SIZE = 1_000_000
N = 25
TRIALS = 10_000


def make_population(seed: int = 0) -> np.ndarray:
    """母集団を作る。母平均・母SDが設計値ちょうどになるよう揃えておく。

    こうしておくと「母平均 50.0000」が丸めの結果ではなく定義になり、標本平均のずれを
    母集団側の誤差と取り違えずに済む。
    """
    raw = np.random.default_rng(seed).normal(MU, SIGMA, size=POP_SIZE)
    return MU + SIGMA * (raw - raw.mean()) / raw.std(ddof=0)


def main() -> None:
    plots.setup()
    pop = make_population(seed=0)
    print("--- 母集団（本来は見えないもの）---")
    print(f"  サイズ N = {pop.size:,}")
    print(f"  母平均 μ = {pop.mean():.4f}   母標準偏差 σ = {pop.std(ddof=0):.4f}")

    def one_sample(rng):
        """母集団から n 人を引いて、その標本平均を返す。"""
        idx = rng.integers(0, pop.size, size=N)   # N が十分大きいので復元抽出でよい
        return float(pop[idx].mean())

    means = sim.repeat(one_sample, trials=TRIALS, seed=5, progress=False)

    print(f"\n--- n={N} の標本を1つ引くと ---")
    for i in range(3):
        print(f"  {i + 1}回目の標本平均 = {means[i]:.2f}")
    print("  ← 同じ母集団から引いたのに、毎回ちがう。標本平均は確率変数である")

    se_theory = SIGMA / np.sqrt(N)
    print(f"\n--- {TRIALS:,}回くり返すと（これが標本分布）---")
    print(f"  標本平均の平均 = {means.mean():.4f}   （母平均 {MU:.4f} のまわりに集まる）")
    print(f"  標本平均のSD   = {means.std(ddof=1):.4f}   （理論 σ/√n = {se_theory:.4f}）")
    print(f"  最小 {means.min():.2f} / 最大 {means.max():.2f}"
          f"   幅 {means.max() - means.min():.2f}")
    within = np.abs(means - MU) <= 1.96 * se_theory
    print(f"  μ ± 1.96·σ/√n に入った割合 = {within.mean():.4f}")

    fig, (ax1, ax2) = plots.figure(1, 2, w=2.0)
    ax1.hist(pop, bins=60, density=True, color=plots.PALETTE["data"], alpha=0.55, lw=0)
    first = pop[np.random.default_rng(99).integers(0, pop.size, size=N)]
    ax1.plot(first, np.full(N, 0.0015), "|", color=plots.PALETTE["estimate"], ms=6)
    plots.mark_truth(ax1, MU, f"母平均 = {MU:g}")
    ax1.set_title(f"母集団（N={pop.size:,}）と 1回ぶんの標本 n={N}")
    ax1.set_xlabel("値")
    ax1.set_ylabel("密度")

    grid = np.linspace(means.min(), means.max(), 400)
    theory = np.exp(-0.5 * ((grid - MU) / se_theory) ** 2) / (se_theory * np.sqrt(2 * np.pi))
    plots.sim_hist(ax2, means, theory=(grid, theory), bins=50,
                   theory_label="N(μ, (σ/√n)²)")
    plots.mark_truth(ax2, MU, f"母平均 = {MU:g}")
    ax2.set_title(f"標本平均の分布（{TRIALS:,}回）")
    ax2.set_xlabel("標本平均")
    ax2.set_ylabel("密度")
    ax2.set_xlim(MU - 4.2 * se_theory, MU + 4.2 * se_theory)
    plots.save(fig, "fig-5-2-statistic-is-random.png")


if __name__ == "__main__":
    main()
