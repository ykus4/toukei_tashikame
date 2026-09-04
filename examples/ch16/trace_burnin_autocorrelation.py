"""提案幅を1つ変えるだけで、同じ 10,000 反復の値打ちが 20 倍変わる。

MCMC の出力は独立な標本ではない。隣どうしが相関しているので、10,000 本引いても
「独立に 10,000 個引いたぶんの情報」は無い。どれだけ目減りしたかを測るのが有効標本数
(ESS) で、目減りの原因は提案幅である。

  - 狭すぎる (0.02): ほぼ全部受容されるが、ちょっとずつしか動かない。受容率は高いのに
    鎖はゆっくり漂うだけで、自己相関が長く残る
  - ちょうどよい (0.5): 受容率 0.2〜0.5、相関はすぐ落ちる
  - 広すぎる (5.0): ほとんど棄却されて同じ場所に留まり続ける。トレースが階段になる

受容率が高いことは良いことではない、というのがこの節の要点である。

    uv run python examples/ch16/trace_burnin_autocorrelation.py
"""

import numpy as np
from scipy import stats

from toukei_tashikame import bayes, plots

MU_TRUE, SIGMA, N = 2.0, 1.0, 50     # 16-2 と同じモデル・同じデータ
SEED = 162
PRIOR_MU, PRIOR_SD = 0.0, 10.0
BURN, KEEP = 1_000, 10_000           # 助走 1,000 を捨てて 10,000 本残す
STEPS = (0.02, 0.5, 5.0)
LAG_SHOWN = 20


def log_posterior(mu: float, y: np.ndarray) -> float:
    """正規化していない事後の対数（16-2 と同一）。"""
    return float(-0.5 * ((mu - PRIOR_MU) / PRIOR_SD) ** 2
                 - 0.5 * np.sum(((y - mu) / SIGMA) ** 2))


def autocorr(chain: np.ndarray, max_lag: int) -> np.ndarray:
    """ラグ 1..max_lag の自己相関。「何ステップで独立になるか」を読む。"""
    c = chain - chain.mean()
    denom = float(c @ c)
    return np.array([float((c[:-k] @ c[k:]) / denom) for k in range(1, max_lag + 1)])


def draw(results, exact_mean, exact_sd) -> None:
    fig, axes = plots.figure(3, 3, w=2.1, h=2.3)
    pal = plots.PALETTE
    grid = np.linspace(exact_mean - 5 * exact_sd, exact_mean + 5 * exact_sd, 400)
    dens = stats.norm.pdf(grid, exact_mean, exact_sd)
    titles = ("狭すぎる", "ちょうどよい", "広すぎる")

    for col, (res, title) in enumerate(zip(results, titles, strict=True)):
        # ① トレース。バーンインも含めて全部描く。
        ax = axes[0][col]
        ax.plot(res["full"], color=pal["posterior"], lw=0.35, zorder=3)
        ax.axvspan(0, BURN, color=pal["reject"], alpha=0.18, lw=0, zorder=1)
        plots.mark_truth(ax, exact_mean, f"共役解 = {exact_mean:.3f}", axis="y")
        ax.set_ylim(exact_mean - 8 * exact_sd, exact_mean + 8 * exact_sd)
        ax.set_title(f"step={res['step']}（{title}）受容率 {res['rate']:.2f}")
        if col == 0:
            ax.set_ylabel("① トレース")

        # ② 自己相関。ここが ESS の正体。
        ax = axes[1][col]
        lags = np.arange(1, 51)
        ax.vlines(lags, 0, res["acf"], color=pal["data"], lw=1.0, zorder=3)
        ax.axhline(0, color=pal["ink2"], lw=0.6)
        ax.axvline(LAG_SHOWN, color=pal["reject"], lw=0.8, ls="--", dashes=(4, 2.2))
        ax.set_ylim(-0.15, 1.05)
        ax.set_xlabel("ラグ")
        ax.set_title(f"lag={LAG_SHOWN} で {res['acf'][LAG_SHOWN - 1]:.2f}")
        if col == 0:
            ax.set_ylabel("② 自己相関")

        # ③ 事後。ESS が小さいほど解析解から外れる。
        ax = axes[2][col]
        ax.hist(res["kept"], bins=45, density=True, color=pal["data"], alpha=0.55, lw=0)
        ax.plot(grid, dens, color=pal["truth"], lw=1.1, ls="--", dashes=(4, 2.0), zorder=5)
        ax.set_xlim(grid[0], grid[-1])
        ax.set_xlabel("$\\mu$")
        ax.set_title(f"ESS {res['ess']:,.0f} / {KEEP:,}")
        if col == 0:
            ax.set_ylabel("③ 事後（赤破線=共役解）")

    fig.tight_layout()
    plots.save(fig, "fig-16-3-trace-and-autocorr.png")


def main() -> None:
    plots.setup()
    y = np.random.default_rng(SEED).normal(MU_TRUE, SIGMA, size=N)
    prec = 1.0 / PRIOR_SD**2 + N / SIGMA**2
    exact_mean = (PRIOR_MU / PRIOR_SD**2 + y.sum() / SIGMA**2) / prec
    exact_sd = float(np.sqrt(1.0 / prec))

    results = []
    for step in STEPS:
        mh = bayes.metropolis_hastings(lambda m: log_posterior(m, y), init=0.0,
                                       n=BURN + KEEP, step=step, seed=SEED + 1)
        kept = mh.burned(BURN)
        results.append({
            "step": step, "rate": mh.accept_rate, "full": mh.chain, "kept": kept,
            "ess": bayes.ess(kept), "acf": autocorr(kept, 50),
            "mean": float(kept.mean()), "sd": float(kept.std(ddof=1)),
            "unique": float(np.unique(kept).size / kept.size),
        })

    print(f"--- 同じモデル・同じデータ（真値 {MU_TRUE}, n={N}, seed={SEED}）を、"
          "提案幅だけ変えて回す ---")
    print(f"    各 {BURN + KEEP:,} 反復、助走 {BURN:,} を捨てて {KEEP:,} 本を使う")
    print(f"    共役解: 事後平均 {exact_mean:.4f}、事後SD {exact_sd:.4f}\n")

    head = f"{'提案幅':>8}{'受容率':>10}{'ESS':>10}{'ESS/本数':>11}" \
           f"{f'自己相関(lag={LAG_SHOWN})':>18}{'事後平均':>11}{'事後SD':>10}"
    print(head)
    for r in results:
        print(f"{r['step']:>8}{r['rate']:>10.4f}{r['ess']:>10,.0f}"
              f"{r['ess'] / KEEP:>11.3f}{r['acf'][LAG_SHOWN - 1]:>18.3f}"
              f"{r['mean']:>11.4f}{r['sd']:>10.4f}")

    best = max(results, key=lambda r: r["ess"])
    worst = min(results, key=lambda r: r["ess"])
    print(f"\n  一番良い step={best['step']} と一番悪い step={worst['step']} で、"
          f"ESS は {best['ess'] / worst['ess']:.1f} 倍ちがう。")
    print(f"  計算時間は同じ {BURN + KEEP:,} 反復である。値打ちだけが違う。\n")

    narrow, wide = results[0], results[2]
    print(f"  step={narrow['step']}: 受容率 {narrow['rate']:.2f} と高いが、"
          "1歩が小さすぎて事後を渡りきれない。")
    print("    受容率の高さは「よく動いている」ではなく「動いていない」の印である。")
    print(f"  step={wide['step']}: 受容率 {wide['rate']:.2f}。棄却されると同じ値が記録"
          f"されるので、{KEEP:,} 本のうち")
    print(f"    異なる値は {wide['unique'] * 100:.1f}% しかない。トレースが階段になる。")
    print("\n  実務では受容率を見て step を調整する。NUTS（16-4 の PyMC の既定）は")
    print("  この調整を自動でやる。手で回すのはここまでで十分である。")

    draw(results, exact_mean, exact_sd)


if __name__ == "__main__":
    main()
