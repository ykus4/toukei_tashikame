"""レイテンシは対数正規。平均が「真ん中」ではないこと、そして n を増やせば適合度検定は必ず棄却すること。

中央値200ms・$\\sigma=0.8$ の対数正規からレイテンシを引く。平均は 275ms 付近に出るが、
その平均を下回るリクエストが全体の3分の2を占める。「平均レイテンシ 275ms」という報告が
体感と食い違うのはこのためで、裾の重い量は分位点（p50 / p95 / p99）で語る。

後半は 4-11 の話。この（明らかに正規でない）データに正規分布を当てはめて KS 適合度検定を
かけると、n=10 ではまず棄却できず、n=100 を超えると確実に棄却される。分布は最初から
最後まで同じで、変わったのは検出力だけである。「p > 0.05 だったから正規と見なせる」が
言えない理由がここにある。

    uv run python examples/ch04/lognormal_mean_and_fit_rejection.py
"""

import numpy as np
from scipy import stats

from toukei_tashikame import datasets, plots

MEDIAN_MS, SIGMA = 200.0, 0.8
N = 100_000
SEED = 15
FIT_NS = (10, 20, 50, 100, 1_000, 10_000)
REPEATS = 200


def ks_against_normal(n: int, rng) -> tuple[np.ndarray, np.ndarray]:
    """対数正規から n 点引いて正規を当てはめ、KS 距離と p 値を REPEATS 本ぶん返す。

    平均と SD をデータから推定してから同じデータで検定する。実務でやる手順そのままで、
    この場合 p 値は本来より大きめに出る（Lilliefors 補正が要る）。
    """
    dists, ps = [], []
    for _ in range(REPEATS):
        s = rng.lognormal(mean=np.log(MEDIAN_MS), sigma=SIGMA, size=n)
        fitted = stats.norm(loc=s.mean(), scale=s.std(ddof=1))
        res = stats.kstest(s, fitted.cdf)
        dists.append(res.statistic)
        ps.append(res.pvalue)
    return np.array(dists), np.array(ps)


def main() -> None:
    x = datasets.latency(N, median_ms=MEDIAN_MS, sigma=SIGMA, seed=SEED)

    print(f"--- 対数正規のレイテンシ（中央値 {MEDIAN_MS:g}ms、σ={SIGMA:g}）を {N:,} 本 ---")
    print(f"  平均     {x.mean():>9.1f} ms   （理論 {MEDIAN_MS * np.exp(SIGMA**2 / 2):.1f}）")
    print(f"  中央値   {np.median(x):>9.1f} ms   （理論 {MEDIAN_MS:.1f}）")
    for q in (0.90, 0.95, 0.99, 0.999):
        theory = MEDIAN_MS * np.exp(SIGMA * stats.norm.ppf(q))
        print(f"  p{q * 100:<6g}{np.quantile(x, q):>9.1f} ms   （理論 {theory:.1f}）")
    print(f"  最大値   {x.max():>9.1f} ms")

    below = float((x < x.mean()).mean())
    print(f"\n  平均以下のリクエストの割合 {below:.4f}"
          f"（理論 Φ(σ/2) = {stats.norm.cdf(SIGMA / 2):.4f}）")
    print(f"  平均 {x.mean():.1f}ms は上位 {100 * (1 - below):.1f}% の側にある。"
          "「平均的なリクエスト」ではない")
    print(f"  p99 は中央値の {np.quantile(x, 0.99) / np.median(x):.1f} 倍。"
          "裾の重い量は平均1つでは要約できない")

    print(f"\n--- 4-11: このデータに正規分布を当てはめて KS 検定（各 {REPEATS} 回）---")
    print(f"{'n':>8}{'KS距離の中央値':>16}{'p値の中央値':>14}{'α=0.05 で棄却した割合':>24}")
    rng = np.random.default_rng(SEED + 1)
    med_p = []
    for n in FIT_NS:
        dists, ps = ks_against_normal(n, rng)
        med_p.append(float(np.median(ps)))
        print(f"{n:>8,}{np.median(dists):>16.4f}{np.median(ps):>14.4f}"
              f"{float((ps < 0.05).mean()):>24.3f}")
    print("  分布は最初から正規ではない。n が小さいうちは「棄却できない」だけで、"
          "適合しているのではない")
    print("  逆に n を十分大きくすれば、どんな実データでも正規性は棄却される。"
          "だから適合度検定の p 値だけで分布を決めない（図と分位点を併せて見る）")
    print("  ※ 平均と SD を当てはめてから同じデータで検定しているので、"
          "この p 値は本来より大きめに出る（Lilliefors 補正が要る）")

    plots.setup()
    fig, axes = plots.figure(1, 2, w=1.0, h=0.85, constrained_layout=True)

    ax = axes[0]
    ax.hist(x[x < 1500], bins=80, density=True, color=plots.PALETTE["data"], alpha=0.55, lw=0)
    grid = np.linspace(1, 1500, 500)
    ax.plot(grid, stats.lognorm.pdf(grid, SIGMA, scale=MEDIAN_MS),
            color=plots.PALETTE["truth"], lw=1.2, ls="--", dashes=(4, 2.0), zorder=5)
    ax.annotate("対数正規の pdf", xy=(400, stats.lognorm.pdf(400, SIGMA, scale=MEDIAN_MS)),
                fontsize=6.0, color=plots.PALETTE["truth"], xytext=(3, 3),
                textcoords="offset points")
    for value, label, color in (
        (np.median(x), f"中央値 {np.median(x):.0f}ms", plots.PALETTE["estimate"]),
        (x.mean(), f"平均 {x.mean():.0f}ms", plots.PALETTE["ink2"]),
        (np.quantile(x, 0.99), f"p99 {np.quantile(x, 0.99):.0f}ms", plots.PALETTE["reject"]),
    ):
        ax.axvline(value, color=color, lw=1.0, ls="-")
        ax.annotate(label, xy=(value, 0.96), xycoords=("data", "axes fraction"),
                    fontsize=5.8, color=color, ha="left", va="top", rotation=90,
                    xytext=(2, 0), textcoords="offset points")
    ax.set_xlabel("レイテンシ (ms)")
    ax.set_ylabel("密度")
    ax.set_title(f"平均以下が {below:.1%}")

    ax = axes[1]
    ax.loglog(FIT_NS, med_p, color=plots.PALETTE["estimate"], lw=1.2, marker="o", ms=2.5)
    ax.axhline(0.05, color=plots.PALETTE["reject"], lw=0.9, ls="--", dashes=(4, 2.2))
    ax.annotate("α = 0.05", xy=(FIT_NS[0], 0.05), fontsize=5.8, color=plots.PALETTE["reject"],
                ha="left", va="bottom")
    ax.set_xlabel("標本サイズ n")
    ax.set_ylabel("KS 検定の p 値（中央値）")
    ax.set_title("n を増やせば必ず棄却される")

    plots.save(fig, "fig-4-7-lognormal-latency.png")


if __name__ == "__main__":
    main()
