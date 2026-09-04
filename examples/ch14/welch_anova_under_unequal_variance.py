"""等分散を仮定した分散分析は、群サイズが不均衡だと α を守らない — Welch なら守る。

通常の分散分析は「全群の分散が等しい」を仮定して、全群をまとめた1つの誤差分散 MSW を
使う。分散が群ごとに違うとき、この共通の物差しは群サイズで重みづけされるので、

  * ばらつきの大きい群が**小さい**とき → MSW が小さく見積もられ、F が大きく出る（甘い）
  * ばらつきの大きい群が**大きい**とき → MSW が大きく見積もられ、F が小さく出る（辛い）

という向きの決まった壊れ方をする。分散比 5:1:1 のまま群サイズだけを入れ替えて
10,000 回ずつ数えると、同じ「等分散の破れ」が、片方では第一種の誤りの増加として、
もう片方では検出力の無駄として現れる。

Welch の分散分析は共通の MSW を使わず、群ごとの分散で重みをつけ、自由度を
Welch–Satterthwaite で調整する。どちらの向きの不均衡でも α のそばに留まる。

    uv run python examples/ch14/welch_anova_under_unequal_variance.py
"""

import numpy as np
from scipy import stats

from toukei_tashikame import plots, sim, testing

SDS = (np.sqrt(5.0), 1.0, 1.0)          # 分散比 5:1:1。真の平均はすべて 0
SIZES_LIBERAL = (15, 15, 45)            # ばらつきの大きい群が小さい
SIZES_CONSERVATIVE = (45, 15, 15)       # ばらつきの大きい群が大きい
SEED, TRIALS, ALPHA = 148, 10_000, 0.05


def classic_f_p(groups: list[np.ndarray]) -> float:
    """通常の一元配置分散分析。全群をまとめた MSW を使う。"""
    allv = np.concatenate(groups)
    grand, k, n = allv.mean(), len(groups), allv.size
    ss_b = sum(a.size * (a.mean() - grand) ** 2 for a in groups)
    ss_w = sum(((a - a.mean()) ** 2).sum() for a in groups)
    f = (ss_b / (k - 1)) / (ss_w / (n - k))
    return float(stats.f.sf(f, k - 1, n - k))


def welch_f_p(groups: list[np.ndarray]) -> float:
    """Welch の分散分析。群ごとの分散で重みをつけ、自由度を調整する。"""
    k = len(groups)
    w = np.array([a.size / a.var(ddof=1) for a in groups])      # 精度が重み
    m = np.array([a.mean() for a in groups])
    mw = (w * m).sum() / w.sum()                                # 重みつき全体平均
    num = (w * (m - mw) ** 2).sum() / (k - 1)
    lam = np.array([(1 - wi / w.sum()) ** 2 / (a.size - 1)
                    for wi, a in zip(w, groups, strict=True)])
    f = num / (1 + 2 * (k - 2) / (k**2 - 1) * lam.sum())
    df2 = (k**2 - 1) / (3 * lam.sum())
    return float(stats.f.sf(f, k - 1, df2))


def one_trial(rng: np.random.Generator, sizes: tuple[int, ...]) -> tuple[float, float]:
    """真の平均差ゼロのデータを作り、(通常のANOVAの p, Welch の p) を返す。"""
    groups = [rng.normal(0.0, sd, size=n) for sd, n in zip(SDS, sizes, strict=True)]
    return classic_f_p(groups), welch_f_p(groups)


def check_against_package() -> None:
    """自作の p が ``testing`` の実装と一致することを、1本のデータで確かめる。"""
    rng = np.random.default_rng(SEED)
    groups = [rng.normal(0.0, sd, size=n)
              for sd, n in zip(SDS, SIZES_LIBERAL, strict=True)]
    ref_c, ref_w = testing.f_oneway(*groups), testing.welch_anova(*groups)
    print("\n  自作の実装と testing モジュールの突き合わせ（データ1本）")
    print(f"    通常のANOVA  自作 p = {classic_f_p(groups):.6f} / "
          f"testing p = {ref_c.pvalue:.6f}（F={ref_c.stat:.4f}, df={ref_c.df}）")
    print(f"    Welch        自作 p = {welch_f_p(groups):.6f} / "
          f"testing p = {ref_w.pvalue:.6f}（F={ref_w.stat:.4f}, "
          f"df=({ref_w.df[0]:.0f}, {ref_w.df[1]:.2f})）")
    print("    ← Welch の分母自由度は整数にならない。データから決まる量だから")


def main() -> None:
    plots.setup()
    var_txt = " : ".join(f"{sd**2:.0f}" for sd in SDS)

    print(f"--- 14-8 分散比 {var_txt}、真の平均差ゼロ、各 {TRIALS:,} 回（seed={SEED}）---")
    print("  群サイズだけを入れ替えて、同じ等分散の破れが逆向きに効くのを見る\n")

    results = {}
    with sim.Timer("14-8 の 20,000 回"):
        for name, sizes in [("大きい分散が小さい群", SIZES_LIBERAL),
                            ("大きい分散が大きい群", SIZES_CONSERVATIVE)]:
            p = sim.repeat(lambda rng, s=sizes: one_trial(rng, s), trials=TRIALS,
                           seed=SEED, progress=False)
            rate = (p < ALPHA).mean(axis=0)
            results[name] = (sizes, rate, np.sqrt(rate * (1 - rate) / TRIALS), p)

    print("\n  群サイズ         SD の割り当て        通常のANOVA        Welch ANOVA")
    for name, (sizes, rate, se, _) in results.items():
        sd_txt = "/".join(f"{sd:.2f}" for sd in SDS)
        print(f"  {'/'.join(map(str, sizes)):<12} {sd_txt:<18} "
              f"{rate[0]:.4f} ±{1.96 * se[0]:.4f}   {rate[1]:.4f} ±{1.96 * se[1]:.4f}"
              f"   {name}")

    lib_rate = results["大きい分散が小さい群"][1]
    con_rate = results["大きい分散が大きい群"][1]
    print(f"\n  名目 α = {ALPHA} に対して")
    print(f"    15/15/45（大きい分散が小さい群に）通常 {lib_rate[0]:.4f} "
          f"= 名目の {lib_rate[0] / ALPHA:.1f} 倍。甘い。有意でないものを有意と言う")
    print(f"    45/15/15（大きい分散が大きい群に）通常 {con_rate[0]:.4f} "
          f"= 名目の {con_rate[0] / ALPHA:.1f} 倍。辛い。守れてはいるが検出力を捨てている")
    print(f"    Welch はどちらでも {lib_rate[1]:.4f} / {con_rate[1]:.4f} で α のそば")
    print("\n  「保守側なら安全」と言えないことに注意。辛い側では、"
          "本当にある差を見逃す確率が上がっている。")
    print("  甘いか辛いかは分散と群サイズの組み合わせで決まるので、"
          "手元のデータを見るまで向きも分からない。")
    print("  等分散の事前検定（Levene など）で選り分ける手はあるが、"
          "二段階にすると全体の α がまた狂う。")
    print("  群サイズが揃わないなら、最初から Welch を既定にするのが単純で安全")

    check_against_package()

    # --- 図 ---
    fig, axes = plots.figure(1, 2, w=1.9, h=1.0)

    ax = axes[0]
    x = np.arange(2)
    width = 0.34
    classic = [lib_rate[0], con_rate[0]]
    welch = [lib_rate[1], con_rate[1]]
    ax.bar(x - width / 2, classic, width, color=plots.PALETTE["reject"], lw=0, zorder=3)
    ax.bar(x + width / 2, welch, width, color=plots.PALETTE["estimate"], lw=0, zorder=3)
    for xi, (c, w) in enumerate(zip(classic, welch, strict=True)):
        ax.annotate(f"{c:.4f}", xy=(xi - width / 2, c), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom", fontsize=6.0,
                    color=plots.PALETTE["reject"])
        # Welch の値は棒の中に置く。名目 α の赤い線とラベルが同じ高さに来るため
        ax.annotate(f"{w:.4f}", xy=(xi + width / 2, w), xytext=(0, -3),
                    textcoords="offset points", ha="center", va="top", fontsize=6.0,
                    color="white", zorder=5)
    plots.mark_truth(ax, ALPHA, f"名目 α = {ALPHA}", axis="y")
    ax.set_xticks(x)
    ax.set_xticklabels(["15/15/45\n大きい分散が小さい群", "45/15/15\n大きい分散が大きい群"])
    ax.set_ylabel("第一種の誤り")
    ax.set_ylim(0, max(classic) * 1.25)
    ax.set_title(f"通常のANOVA（橙）と Welch（青）  分散比 {var_txt}")

    ax = axes[1]
    p_lib = results["大きい分散が小さい群"][3]
    grid = np.linspace(0, 1, 201)
    for col, label, series in [(plots.PALETTE["reject"], "通常のANOVA", p_lib[:, 0]),
                               (plots.PALETTE["estimate"], "Welch ANOVA", p_lib[:, 1])]:
        ecdf = np.searchsorted(np.sort(series), grid, side="right") / series.size
        ax.plot(grid, ecdf, color=col, lw=1.3, zorder=4)
        ax.annotate(label, xy=(0.62, np.interp(0.62, grid, ecdf)), xytext=(0, -3),
                    textcoords="offset points", ha="left", va="top", fontsize=6.2, color=col)
    ax.plot(grid, grid, color=plots.PALETTE["truth"], lw=1.0, ls="--", dashes=(4, 2.0), zorder=5)
    ax.annotate("一様分布（正しい p 値）", xy=(0.30, 0.30), xytext=(3, -4),
                textcoords="offset points", ha="left", va="top", fontsize=6.0,
                color=plots.PALETTE["truth"])
    ax.axvline(ALPHA, color=plots.PALETTE["ink2"], lw=0.8, ls=":", zorder=3)
    ax.set_xlabel("p 値")
    ax.set_ylabel("累積割合")
    ax.set_title("15/15/45 での p 値の累積分布")
    fig.tight_layout()
    plots.save(fig, "fig-14-8-welch-vs-classic-anova.png")


if __name__ == "__main__":
    main()
