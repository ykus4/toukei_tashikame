"""ブートストラップ — 標本を母集団に見立てて引き直す。3つの区間の作り方を比べる。

n=40 の対数正規標本（真の平均 e = 2.7183）に B=10,000 のブートストラップを当て、
percentile / basic / BCa の3つの区間を作る。式を解かずに区間が出てくるのがこの方法の
強みで、弱みは「標本が母集団の縮図である」という仮定にすべてを賭けている点にある。

3つは同じ再標本から作るのに、置き場所が違う。歪んだ統計量では basic が下に、BCa が上に
ずれる。どれが名目の95%に近いかは被覆を数えれば分かる——n=40 の対数正規では、どれも
95% には届かない。ブートストラップは中心極限定理の代わりにはならない。

    uv run python examples/ch05/bootstrap_three_intervals.py
"""

import numpy as np

from toukei_tashikame import estimate, plots, sim

SIGMA_LOG = 1.0
MU_LOG = 0.5
TRUE_MEAN = float(np.exp(MU_LOG + 0.5 * SIGMA_LOG**2))   # = e = 2.7183
N = 40
B_SHOW = 10_000      # 1本の標本を詳しく見るとき
B_COVER = 600        # 被覆を数えるとき（B×反復回数が効くので落とす）
REPEATS = 1_000
KINDS = ["percentile", "basic", "bca"]
SEED = 21


def main() -> None:
    plots.setup()
    x = np.random.default_rng(SEED).lognormal(MU_LOG, SIGMA_LOG, size=N)
    print(f"--- n={N} の対数正規標本（真の平均 {TRUE_MEAN:.4f}）---")
    print(f"  標本平均 {x.mean():.4f} / 標本SD {x.std(ddof=1):.4f}"
          f" / 中央値 {np.median(x):.4f} / 最大 {x.max():.4f}")

    boots = estimate.bootstrap(x, B=B_SHOW, seed=SEED)
    print(f"  再標本 {B_SHOW:,} 個の平均のSD = {boots.std(ddof=1):.4f}"
          f"   （標本から出した SE = {x.std(ddof=1) / np.sqrt(N):.4f}）")

    print(f"\n--- 3つの区間（B={B_SHOW:,}）---")
    shown = {}
    for kind in KINDS:
        lo, hi = estimate.boot_ci(x, B=B_SHOW, kind=kind, seed=SEED)
        shown[kind] = (lo, hi)
        hit = "包んだ" if lo <= TRUE_MEAN <= hi else "外した"
        print(f"  {kind:<11} [{lo:.2f}, {hi:.2f}]  幅 {hi - lo:.2f}   真値を{hit}")
    lo_t, hi_t = estimate.ci_mean_t(x)
    print(f"  {'（参考）t 区間':<9} [{lo_t:.2f}, {hi_t:.2f}]  幅 {hi_t - lo_t:.2f}")

    def interval_fn(kind: str):
        def one(rng):
            sample = rng.lognormal(MU_LOG, SIGMA_LOG, size=N)
            return estimate.boot_ci(sample, B=B_COVER, kind=kind,
                                    seed=int(rng.integers(2**31 - 1)))
        return one

    print(f"\n--- 被覆を数える（B={B_COVER:,} × {REPEATS:,}回）---")
    with sim.Timer("  所要"):
        cover = {}
        for kind in KINDS:
            # seed を揃えるので、3つの区間は同じ標本の列から作られる（対応のある比較）
            cover[kind] = sim.coverage(interval_fn(kind), truth=TRUE_MEAN,
                                       trials=REPEATS, seed=700, progress=False)
    for kind in KINDS:
        res = cover[kind]
        iv = res.intervals
        low = float((iv[:, 1] < TRUE_MEAN).mean())
        high = float((iv[:, 0] > TRUE_MEAN).mean())
        print(f"  {kind:<11} 被覆 {res.rate:.4f} ± {1.96 * res.se:.4f}"
              f"   下に外す {low:.4f} / 上に外す {high:.4f}"
              f"   平均幅 {np.mean(iv[:, 1] - iv[:, 0]):.3f}")
    print("  ← 名目は 95%。BCa がいちばん近いが、それでも届かない。"
          "n=40 で右に裾を引く母集団では\n     どの作り方も「標本が母集団の縮図」から"
          "外れたぶんだけ損をする")

    fig, (ax1, ax2) = plots.figure(1, 2, w=2.0)
    plots.sim_hist(ax1, boots, bins=50, label="再標本の平均")
    plots.mark_truth(ax1, TRUE_MEAN, f"真の平均 = {TRUE_MEAN:.2f}")
    ax1.axvline(x.mean(), color=plots.PALETTE["estimate"], lw=1.0, ls="--",
                dashes=(4, 2.0))
    ax1.annotate("標本平均", xy=(x.mean(), 0.60), xycoords=("data", "axes fraction"),
                 fontsize=6.0, color=plots.PALETTE["estimate"], ha="right",
                 xytext=(-3, 0), textcoords="offset points")
    for i, kind in enumerate(KINDS):
        lo, hi = shown[kind]
        y = 0.30 - 0.07 * i
        ax1.plot([lo, hi], [y, y], transform=ax1.get_xaxis_transform(),
                 color=plots.PALETTE["interval"], lw=1.8, solid_capstyle="butt")
        ax1.annotate(kind, xy=(hi, y), xycoords=("data", "axes fraction"),
                     xytext=(3, -2), textcoords="offset points", fontsize=6.0,
                     color=plots.PALETTE["estimate"])
    ax1.set_title(f"ブートストラップ分布と3つの区間（B={B_SHOW:,}）")
    ax1.set_xlabel("平均")
    ax1.set_ylabel("密度")

    rates = [cover[k].rate for k in KINDS]
    errs = [1.96 * cover[k].se for k in KINDS]
    ax2.bar(KINDS, rates, yerr=errs, color=plots.PALETTE["estimate"], width=0.55,
            capsize=3, ecolor=plots.PALETTE["ink2"])
    plots.mark_truth(ax2, 0.95, "名目 95%", axis="y")
    ax2.set_ylim(0.80, 1.0)
    ax2.set_ylabel("実際の被覆確率")
    ax2.set_title(f"被覆（{REPEATS}回, 誤差棒は ±1.96SE）")
    plots.save(fig, "fig-5-7-bootstrap-intervals.png")


if __name__ == "__main__":
    main()
