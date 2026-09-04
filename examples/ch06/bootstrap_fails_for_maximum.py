"""ブートストラップが効かない例 — 平均なら当たるが、最大値では当たらない。

一様分布 U(0, 100) の上限 θ=100 を n=50 の標本から推す。percentile ブートストラップ
区間を 1,000 回作って被覆を数えると、同じ標本・同じ再標本で作った平均の区間が名目どおり
当たる横で、最大値の区間は一度も当たらない。

理由は再標本の作り方にある。復元抽出で引き直しても、手元にない値は決して現れない。
再標本の最大値は必ず標本最大以下で、標本最大は必ず θ 未満である。区間の上端が θ を
超えられない以上、被覆は 0 にしかならない。ブートストラップは「標本が母集団の縮図」を
仮定するが、分布の端はいちばん縮図になりにくい場所である。

    uv run python examples/ch06/bootstrap_fails_for_maximum.py
"""

import numpy as np

from toukei_tashikame import plots, sim

THETA, MU_TRUE, N = 100.0, 50.0, 50
B, TRIALS, SEED = 1_000, 1_000, 30
SHOW_SEED = 31   # 図に描く「1つの標本」用


def one_trial(rng) -> tuple[float, ...]:
    """1標本ぶん。最大値と平均に、同じ再標本から percentile 区間を作る。"""
    x = rng.uniform(0.0, THETA, size=N)
    boot = x[rng.integers(0, N, size=(B, N))]
    max_lo, max_hi = np.quantile(boot.max(axis=1), [0.025, 0.975])
    mean_lo, mean_hi = np.quantile(boot.mean(axis=1), [0.025, 0.975])
    # 比較用: 最大値の分布は手で解ける。max/θ ~ Beta(n,1) を反転した厳密な区間。
    exact_lo = x.max() / 0.975 ** (1 / N)
    exact_hi = x.max() / 0.025 ** (1 / N)
    return max_lo, max_hi, mean_lo, mean_hi, float(x.max()), exact_lo, exact_hi


def rate(hits: np.ndarray) -> str:
    """割合と、数え直しの誤差を並べて返す。"""
    r = float(hits.mean())
    return f"{r:.4f} ± {1.96 * np.sqrt(r * (1 - r) / hits.size):.4f}"


def main() -> None:
    plots.setup()
    out = sim.repeat(one_trial, trials=TRIALS, seed=SEED, progress=False)
    max_lo, max_hi, mean_lo, mean_hi, smax, ex_lo, ex_hi = out.T

    cov_max = (max_lo <= THETA) & (THETA <= max_hi)
    cov_mean = (mean_lo <= MU_TRUE) & (MU_TRUE <= mean_hi)
    cov_exact = (ex_lo <= THETA) & (THETA <= ex_hi)

    print(f"--- U(0, {THETA:g}) から n={N} を引く、を {TRIALS:,} 回"
          f"（B={B:,} の percentile ブートストラップ）---")
    print(f"  {'推す量':<26}{'真値':>8}{'被覆':>20}{'平均幅':>10}")
    print(f"  {'最大 θ（bootstrap）':<26}{THETA:>8.1f}{rate(cov_max):>20}"
          f"{np.mean(max_hi - max_lo):>10.2f}")
    print(f"  {'平均 μ（bootstrap）':<26}{MU_TRUE:>8.1f}{rate(cov_mean):>20}"
          f"{np.mean(mean_hi - mean_lo):>10.2f}")
    print(f"  {'最大 θ（厳密な区間）':<26}{THETA:>8.1f}{rate(cov_exact):>20}"
          f"{np.mean(ex_hi - ex_lo):>10.2f}")
    print("  同じ標本・同じ再標本を使っている。違うのは「何を推したか」だけ")

    print("\n--- なぜ当たらないか ---")
    capped = int(np.sum(np.isclose(max_hi, smax)))
    print(f"  区間の上端が標本最大とぴったり同じ  {capped:,} / {TRIALS:,} 回")
    print(f"  標本最大の平均                     {smax.mean():.4f}"
          f"（理論 n/(n+1)·θ = {N / (N + 1) * THETA:.4f}）")
    print(f"  区間の上端の平均                   {max_hi.mean():.4f}"
          f"   ← 真値 {THETA:g} に一度も届かない（最大でも {max_hi.max():.4f}）")
    print(f"  区間が真値より下 {int(np.sum(max_hi < THETA)):,} 回 / "
          f"真値より上 {int(np.sum(max_lo > THETA)):,} 回"
          "   ← 外れ方が片側に全振りされている")
    hit_prob = 1 - (1 - 1 / N) ** N
    print(f"  1つの再標本が標本最大を含む確率 1-(1-1/n)^n = {hit_prob:.4f}。"
          "97.5% 点が標本最大に張り付くにはこれで十分すぎる")

    print("\n--- 直し方 ---")
    print(f"  max/θ ~ Beta(n, 1) を反転すると [max/0.975^(1/n), max/0.025^(1/n)]。"
          f"被覆 {rate(cov_exact)}")
    print("  端の推定は分布の形そのものに依存する。再標本に頼らず、手で解けるなら解く")

    # 図: 1つの標本で、再標本の分布がどこに乗るかを見る
    rng = np.random.default_rng(SHOW_SEED)
    x = rng.uniform(0.0, THETA, size=N)
    boot = x[rng.integers(0, N, size=(B, N))]
    fig, axes = plots.figure(1, 2, w=2.0)
    panels = (
        ("最大値 — 当たらない", boot.max(axis=1), THETA, "θ", float(x.max())),
        ("平均 — 当たる", boot.mean(axis=1), MU_TRUE, "μ", float(x.mean())),
    )
    for ax, (title, boots, truth, sym, point) in zip(axes, panels, strict=True):
        lo, hi = np.quantile(boots, [0.025, 0.975])
        ax.hist(boots, bins=40, density=True, color=plots.PALETTE["data"], alpha=0.55, lw=0)
        plots.mark_interval(ax, lo, hi, label=f"[{lo:.1f}, {hi:.1f}]")
        ax.axvline(point, color=plots.PALETTE["estimate"], lw=1.1)
        ax.annotate(f"推定値 {point:.1f}", xy=(point, 0.72), xycoords=("data", "axes fraction"),
                    ha="right", va="top", fontsize=6.0, color=plots.PALETTE["estimate"],
                    xytext=(-3, 0), textcoords="offset points")
        plots.mark_truth(ax, truth, f"真の {sym} = {truth:g}")
        ax.set_xlabel(f"再標本の{sym}の推定値（B={B:,}）")
        ax.set_ylabel("密度")
        ax.set_title(title)
    axes[0].set_xlim(min(x.max() - 12, THETA - 12), THETA + 2)
    plots.save(fig, "fig-6-8-bootstrap-max-fails.png")


if __name__ == "__main__":
    main()
