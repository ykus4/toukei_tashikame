"""中心極限定理の破れ目 — 「n≥30 なら正規」を信じると被覆が 95% に届かない。

σ=1.5 の対数正規（真の平均 exp(1.125)=3.0802）から n=30 を引いて t 信頼区間を作る、を
10,000回。名目 95% の区間が実際に真値を包む割合を数える。

落ち方だけでなく**外れ方の非対称**を見る。歪んだ母集団では標本平均が真値より下に出る
ことが多く、しかも標準偏差も同時に過小に出るため、区間は「低いところに、狭く」できる。
外しは下側に偏り、n を増やしてもこの偏りはゆっくりとしか消えない。

    uv run python examples/ch05/clt_fails_for_lognormal_n30.py
"""

import unicodedata

import numpy as np

from toukei_tashikame import estimate, plots, sim

SIGMA_LOG = 1.5
TRUE_MEAN = float(np.exp(0.5 * SIGMA_LOG**2))   # 対数正規の平均 exp(μ + σ²/2)
N_LIST = [30, 100, 1_000]
TRIALS = 10_000


def interval_fn(n: int):
    """n の対数正規標本から 95% t 区間を作る試行を返す。"""
    def one(rng):
        x = rng.lognormal(mean=0.0, sigma=SIGMA_LOG, size=n)
        return estimate.ci_mean_t(x, conf=0.95)
    return one


def rj(text: str, width: int) -> str:
    """全角を2桁として数えて右詰めする。日本語の見出しでも表の桁が揃う。"""
    w = sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in text)
    return " " * max(width - w, 0) + text


def main() -> None:
    plots.setup()
    print(f"--- 対数正規（σ={SIGMA_LOG}）の真の平均 = {TRUE_MEAN:.4f} ---")
    print(f"  名目 95% の t 信頼区間を各 {TRIALS:,}回作って、包んだ割合を数える\n")
    print("  " + rj("n", 6) + "  " + rj("被覆", 8) + "  " + rj("±1.96SE", 8)
          + "  " + rj("下に外す", 9) + "  " + rj("上に外す", 9) + "  " + rj("区間の平均幅", 12))

    results = {}
    for i, n in enumerate(N_LIST):
        res = sim.coverage(interval_fn(n), truth=TRUE_MEAN, trials=TRIALS,
                           seed=500 + i, progress=False)
        lo, hi = res.intervals[:, 0], res.intervals[:, 1]
        miss_low = float((hi < TRUE_MEAN).mean())    # 区間がまるごと真値の下
        miss_high = float((lo > TRUE_MEAN).mean())   # まるごと上
        width = float((hi - lo).mean())
        results[n] = (res, miss_low, miss_high, width)
        print(f"  {n:>6}  {res.rate:>8.4f}  {1.96 * res.se:>8.4f}  {miss_low:>9.4f}"
              f"  {miss_high:>9.4f}  {width:>12.4f}")

    res30, low30, high30, _ = results[30]
    print(f"\n  n=30 では 95% のはずが {res30.rate:.4f}。足りない {0.95 - res30.rate:.4f} は"
          "「たまたま」では説明できない")
    print(f"  外し方も対称でない: 下に {low30:.4f} / 上に {high30:.4f}"
          f"（{low30 / high30:.1f} 倍）")
    print(f"  n={N_LIST[-1]:,} でも {results[N_LIST[-1]][0].rate:.4f}。"
          "歪んだ母集団では n≥30 は目安として弱すぎる")

    fig, (ax1, ax2) = plots.figure(1, 2, w=2.0)
    missed = plots.coverage_stripes(ax1, res30.intervals, TRUE_MEAN, n_show=100)
    ax1.set_title(f"n=30 の区間 100本（赤 = 外した {missed} 本）")
    ax1.set_xlabel("平均の推定区間")
    ax1.set_xlim(0, 9)

    ns = np.array(N_LIST, dtype=float)
    rates = np.array([results[n][0].rate for n in N_LIST])
    ses = np.array([1.96 * results[n][0].se for n in N_LIST])
    ax2.errorbar(ns, rates, yerr=ses, fmt="o-", ms=3, lw=1.0,
                 color=plots.PALETTE["estimate"], ecolor=plots.PALETTE["estimate"],
                 capsize=2)
    plots.mark_truth(ax2, 0.95, "名目 95%", axis="y")
    ax2.set_xscale("log")
    ax2.set_ylim(0.75, 0.98)
    ax2.set_xlabel("標本サイズ n（対数）")
    ax2.set_ylabel("実際の被覆確率")
    ax2.set_title(f"名目に届かない（各 {TRIALS:,}回）")
    plots.save(fig, "fig-5-6-lognormal-undercoverage.png")


if __name__ == "__main__":
    main()
