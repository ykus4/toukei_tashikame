"""独立の仮定を1つだけ壊すと、5%のはずの誤りが何%になるか。

検定は「観測が互いに独立」を仮定している。この仮定は正規性とちがって図では見えず、
壊れていても t 値は素知らぬ顔で出てくる。ここでは平均も分散も正規性もそのままにして、
**隣り合う観測が相関している**という一点だけを変える（AR(1)、ρ=0.6、n=50）。

第一種の誤りは 5% から動く。動くのは、隣どうしが似ていると n=50 個のデータが実質
50 個ぶんの情報を持たないためで、標準誤差が小さく見積もられる。時系列やクラスタの
効いたログデータに素朴な t 検定を当てると、この形で有意が量産される。

    uv run python examples/ch03/independence_broken_correlated_data.py
"""

import numpy as np
from scipy import signal, stats

from toukei_tashikame import plots, sim

N = 50
RHO = 0.6
TRIALS = 10_000
ALPHA = 0.05
SEED = 5


def ar1(rng: np.random.Generator, rho: float, n: int = N) -> np.ndarray:
    """定常な AR(1)。ρ=0 なら独立、どの ρ でも平均0・分散1に揃えてある。

    x[t] = ρ x[t-1] + ε[t]、ε ~ N(0, 1-ρ²)。分散を揃えておかないと、第一種の誤りが
    動いた原因が「相関」なのか「ばらつきの大きさ」なのか分からなくなる。
    """
    eps = rng.normal(0.0, np.sqrt(1.0 - rho**2), size=n)
    x0 = rng.normal(0.0, 1.0)                       # 初期値も定常分布から引く
    return signal.lfilter([1.0], [1.0, -rho], eps, zi=[rho * x0])[0]


def pvalue_of_ttest(rng: np.random.Generator, rho: float) -> float:
    """μ=0 が正しいデータに1標本 t 検定を当て、p 値を返す（帰無は常に真）。"""
    x = ar1(rng, rho)
    t = x.mean() / (x.std(ddof=1) / np.sqrt(N))
    return float(2 * stats.t.sf(abs(t), df=N - 1))


def main() -> None:
    plots.setup()

    print(f"--- μ=0 のデータに1標本t検定。帰無は常に真、n={N}、{TRIALS:,}回 ---")
    indep = sim.rejection_rate(lambda r: pvalue_of_ttest(r, 0.0),
                               alpha=ALPHA, trials=TRIALS, seed=SEED, progress=False)
    dep = sim.rejection_rate(lambda r: pvalue_of_ttest(r, RHO),
                             alpha=ALPHA, trials=TRIALS, seed=SEED, progress=False)
    print(f"  独立（ρ=0.0）    第一種の誤り {indep}")
    print(f"  AR(1)（ρ={RHO}）  第一種の誤り {dep}")
    print(f"  → 名目 α={ALPHA} の {dep.rate / ALPHA:.1f} 倍。"
          "有意水準は宣言した値ではなく、仮定が成り立ったときの値でしかない")

    print("\n--- ρ を掃く（各2,000回）---")
    rhos = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    print(f"  {'ρ':>4}  {'第一種の誤り':>12}  {'実効サンプルサイズ':>18}")
    rates = []
    for i, rho in enumerate(rhos):
        res = sim.rejection_rate(lambda r, rho=rho: pvalue_of_ttest(r, rho),
                                 alpha=ALPHA, trials=2_000, seed=SEED + 1000 * (i + 1),
                                 progress=False)
        rates.append(res.rate)
        # 相関があるときの実効サンプルサイズ n(1-ρ)/(1+ρ)。50個が何個ぶんに見えるか。
        n_eff = N * (1 - rho) / (1 + rho)
        print(f"  {rho:>4.1f}  {res.rate:>12.4f}  {n_eff:>14.1f} 個")

    print(f"\n  ρ=0.9 では {N} 個のデータが実質 {N * 0.1 / 1.9:.1f} 個ぶん。"
          "それを50個として扱うから標準誤差が小さく出て、t が大きくなる")

    fig, (ax1, ax2) = plots.figure(1, 2, w=2.0, h=0.95)

    # 左: 系列そのもの。独立とAR(1)を重ねると「似ている隣」が見える。
    rng = np.random.default_rng(SEED)
    ax1.plot(ar1(rng, 0.0), color=plots.PALETTE["data"], lw=0.9, zorder=3)
    ax1.plot(ar1(rng, RHO), color=plots.PALETTE["alt"], lw=1.1, zorder=4)
    ax1.annotate("独立（ρ=0）", xy=(0.03, 0.05), xycoords="axes fraction",
                 fontsize=6.2, color=plots.PALETTE["data"])
    ax1.annotate(f"AR(1)（ρ={RHO}）", xy=(0.03, 0.93), xycoords="axes fraction",
                 va="top", fontsize=6.2, color=plots.PALETTE["alt"])
    ax1.margins(y=0.22)   # 注記を置く余白
    ax1.set_xlabel("観測の順番")
    ax1.set_ylabel("値")
    ax1.set_title("平均も分散も同じ。違うのは隣との関係だけ")

    # 右: ρ ごとの第一種の誤り。名目 5% の水平線からどれだけ離れるか。
    ax2.plot(rhos, rates, "o-", color=plots.PALETTE["estimate"], ms=3, lw=1.0, zorder=4)
    ax2.fill_between(rhos, ALPHA, rates, color=plots.PALETTE["reject"], alpha=0.30, lw=0,
                     zorder=2)   # 名目 α からはみ出した分＝余分に出る誤り
    plots.mark_truth(ax2, ALPHA, "名目 α = 0.05", axis="y")
    ax2.set_xlabel("隣どうしの相関 ρ")
    ax2.set_ylabel("第一種の誤り（実測）")
    ax2.set_title("独立を壊すと α が膨らむ")

    fig.tight_layout()
    plots.save(fig, "fig-3-5-dependence-inflates-alpha.png")


if __name__ == "__main__":
    main()
