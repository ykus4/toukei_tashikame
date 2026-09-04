"""事前分布の影響は、データが増えれば消える。消えるまでの速さを測る。

「ベイズは事前分布で結論を操作できる」という批判は、n が小さいときには正しい。
だが事前は「すでに何件見たか」に翻訳できる重さでしかなく、$\\mathrm{Beta}(20, 80)$ は
100 件ぶんの重さしかない。1,000 件のデータの前では 1/10 の発言力に落ちる。

真値 0.30 のデータを、無情報 $\\mathrm{Beta}(1,1)$・弱情報 $\\mathrm{Beta}(2,2)$・
真値から外れた強い事前 $\\mathrm{Beta}(20,80)$（平均 0.20）の3つで更新し、事後平均が
どこで一致するかを数える。事前を隠さずに書けばよい、という結論のための節である。

    uv run python examples/ch15/prior_influence_vanishes_with_n.py
"""

import numpy as np

from toukei_tashikame import bayes, plots

P_TRUE, SEED = 0.30, 156
N_MAX = 10_000
NS = (1, 3, 10, 30, 100, 300, 1000, 3000, 10_000)

# (名前, a, b)。a + b - 2 が「すでに見た件数」に相当する重さ。
PRIORS = (
    ("無情報 Beta(1,1)", 1.0, 1.0),
    ("弱情報 Beta(2,2)", 2.0, 2.0),
    ("強い誤り Beta(20,80)", 20.0, 80.0),
)


def posteriors_at(stream: np.ndarray, n: int) -> list[bayes.BetaPosterior]:
    k = int(stream[:n].sum())
    return [bayes.beta_binomial(k, n, a, b) for _, a, b in PRIORS]


def draw(stream: np.ndarray, ns: np.ndarray) -> None:
    fig, axes = plots.figure(1, 2, w=2.0, h=0.95)
    pal = plots.PALETTE
    styles = (
        (pal["prior"], (4, 2.0)),
        (pal["estimate"], None),
        (pal["alt"], (1.5, 1.5)),
    )

    means = np.array([[p.mean for p in posteriors_at(stream, int(n))] for n in ns])

    ax = axes[0]
    for j, ((name, _, _), (color, dashes)) in enumerate(zip(PRIORS, styles, strict=True)):
        kw = {"ls": "--", "dashes": dashes} if dashes else {}
        ax.plot(ns, means[:, j], color=color, lw=1.2, zorder=4, **kw)
        ax.annotate(name, xy=(0.40, 0.93 - 0.09 * j), xycoords="axes fraction",
                    fontsize=5.8, color=color)
    plots.mark_truth(ax, P_TRUE, f"真値 = {P_TRUE}", axis="y")
    ax.set_xscale("log")
    ax.set_xlabel("観測件数 n（対数）")
    ax.set_ylabel("事後平均")
    ax.set_title("① 3本はやがて1本に重なる")

    ax = axes[1]
    spread = means.max(axis=1) - means.min(axis=1)
    ax.plot(ns, spread, color=pal["posterior"], lw=1.3, zorder=4)
    anchor = int(np.argmin(np.abs(ns - 100)))          # n=100 の実測に合わせて引く
    ref = spread[anchor] * ns[anchor] / ns
    ax.plot(ns, ref, color=pal["truth"], lw=1.0, ls="--", dashes=(4, 2.0), zorder=3)
    ax.annotate("$1/n$ の傾き\n（n=100 で合わせた）", xy=(ns[-1], ref[-1]), xytext=(-6, 24),
                ha="right", textcoords="offset points", fontsize=6.0, color=pal["truth"])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("観測件数 n（対数）")
    ax.set_ylabel("事後平均の最大差")
    ax.set_title("② 事前による差は $1/n$ で消える")

    plots.save(fig, "fig-15-6-prior-washes-out.png")


def main() -> None:
    plots.setup()
    rng = np.random.default_rng(SEED)
    stream = (rng.random(N_MAX) < P_TRUE).astype(int)   # 1本の列を先頭から伸ばす

    print(f"--- 真値 {P_TRUE}、同じ列（seed={SEED}）を3つの事前で更新 ---\n")
    print("  事前は「すでに何件見たか」に翻訳できる:")
    for name, a, b in PRIORS:
        print(f"    {name:<22} 平均 {a / (a + b):.2f}、重さ {a + b - 2:.0f} 件ぶん")
    print()

    header = "".join(f"{name:>22}" for name, _, _ in PRIORS)
    print(f"{'n':>8}{'k':>7}{header}{'最大差':>10}")
    for n in NS:
        posts = posteriors_at(stream, n)
        cells = "".join(f"{p.mean:>22.4f}" for p in posts)
        spread = max(p.mean for p in posts) - min(p.mean for p in posts)
        print(f"{n:>8,}{posts[0].k:>7}{cells}{spread:>10.4f}")

    small = posteriors_at(stream, 10)
    big = posteriors_at(stream, 10_000)
    print(f"\n  n=10 では観測 CVR が {small[0].k / 10:.2f} なのに、強い事前 Beta(20,80) の"
          f"事後平均は {small[2].mean:.4f}。")
    print(f"  データ 10 件に対して事前が 100 件ぶんの重さを持つので、"
          f"事後は事前の平均 {20 / 100:.2f} 側に引かれたままである。")
    print(f"  n=10,000 では3つとも {big[0].mean:.4f} 付近に集まり、最大差は "
          f"{max(p.mean for p in big) - min(p.mean for p in big):.4f} しかない。\n")

    spreads = {n: (lambda ps: max(p.mean for p in ps) - min(p.mean for p in ps))(
        posteriors_at(stream, n)) for n in (10, 100, 1000, 10_000)}
    print(f"  差の縮み方: n=10 {spreads[10]:.4f} → n=100 {spreads[100]:.4f} → "
          f"n=1000 {spreads[1000]:.4f} → n=10000 {spreads[10_000]:.4f}")
    print(f"  n を 100 倍した区間（100 → 10,000）で {spreads[100] / spreads[10_000]:.0f} 分の1。")
    print("  差が $1/\\sqrt{n}$ ではなく $1/n$ で消えるのは、事前が足す (a, b) が n によらず")
    print("  一定で、分母の a+b だけが n とともに伸びるから。小さい n で比が暴れるのは、")
    print("  そこでは k のばらつき自体が大きいためで、右の図でも n<100 は直線に乗らない。\n")

    print("だから実務では、事前を隠すのではなく書く。「Beta(20,80) を使った」と明記すれば、")
    print("読み手はそれが 100 件ぶんの主張だと分かり、手元の n と比べて重みを判断できる。")
    print("事前が結論を決めているなら、それはデータが足りないという事実の別の言い方である。")

    ns = np.unique(np.round(np.logspace(0, np.log10(N_MAX), 60)).astype(int))
    draw(stream, ns)


if __name__ == "__main__":
    main()
