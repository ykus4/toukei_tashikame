"""小さい標本では、「Bが勝つ確率」は事前分布だけで大きく動く。

$\\Pr[p_B > p_A]$ は意思決定にそのまま使える形をしているが、その形をしているからこそ
危ない。各群 12 件しかないとき、この数字は事前をどう置いたかで 0.61 にも 0.86 にもなり、切り替えの判断が反転する。
データは1文字も変わっていない。

同じデータ列を 1,200 件まで伸ばすと3つの事前は同じ答えに収束する。つまりこれは
ベイズの欠陥ではなく、**12 件では何も分かっていない**という事実が事前の違いとして
表に出ているだけである。早期に打ち切りたくなったときに読み返す節。

    uv run python examples/ch15/prior_flips_the_conclusion_small_n.py
"""

import numpy as np

from toukei_tashikame import bayes, datasets, plots

N_FULL, N_SMALL = 1200, 12
P_A, LIFT, SEED = 0.10, 0.20, 159       # 真値は A 10.0% / B 12.0%
DRAWS = 200_000
SHIP_THRESHOLD = 0.80                   # 「Bに切り替える」と決める閾値

# (名前, a, b, 説明)
PRIORS = (
    ("一様 Beta(1,1)", 1.0, 1.0, "何も知らない"),
    ("過去実績 Beta(30,270)", 30.0, 270.0, "CVR 10% を 300 件ぶんの重さで"),
    ("Jeffreys Beta(.5,.5)", 0.5, 0.5, "変換に不変な無情報事前"),
)


def prob_b_wins(d, n: int, a: float, b: float) -> float:
    """先頭 n 件だけを見たときの Pr[p_B > p_A]。"""
    post_a = bayes.beta_binomial(int(d.a[:n].sum()), n, a, b)
    post_b = bayes.beta_binomial(int(d.b[:n].sum()), n, a, b)
    return bayes.prob_b_beats_a(post_a, post_b, draws=DRAWS, seed=SEED)


def draw(d, ns: np.ndarray, curves: np.ndarray) -> None:
    fig, axes = plots.figure(1, 2, w=2.0, h=0.95)
    pal = plots.PALETTE
    styles = ((pal["prior"], (4, 2.0)), (pal["alt"], (1.5, 1.5)), (pal["posterior"], None))
    grid = np.linspace(0.0, 0.8, 600)

    ax = axes[0]
    for j, ((name, a, b, _), (color, dashes)) in enumerate(zip(PRIORS, styles, strict=True)):
        kw = {"ls": "--", "dashes": dashes} if dashes else {}
        post_b = bayes.beta_binomial(int(d.b[:N_SMALL].sum()), N_SMALL, a, b)
        ax.plot(grid, post_b.pdf(grid), color=color, lw=1.2, zorder=4, **kw)
        ax.annotate(name, xy=(0.42, 0.92 - 0.10 * j), xycoords="axes fraction",
                    fontsize=5.8, color=color)
    plots.mark_truth(ax, P_A * (1 + LIFT), f"B の真値 = {P_A * (1 + LIFT):.2f}")
    ax.set_xlabel("$p_B$")
    ax.set_ylabel("事後密度")
    ax.set_title(f"① n={N_SMALL} での B の事後（同じ {int(d.b[:N_SMALL].sum())}/{N_SMALL}）")

    ax = axes[1]
    for j, (color, dashes) in enumerate(styles):
        kw = {"ls": "--", "dashes": dashes} if dashes else {}
        ax.plot(ns, curves[:, j], color=color, lw=1.2, zorder=4, **kw)
    ax.axhline(SHIP_THRESHOLD, color=pal["reject"], lw=1.0, ls="--", dashes=(4, 2.0), zorder=3)
    ax.annotate(f"切り替え閾値 {SHIP_THRESHOLD:.2f}", xy=(ns[-1], SHIP_THRESHOLD),
                xytext=(-3, -10), ha="right", textcoords="offset points", fontsize=5.8,
                color=pal["reject"])
    ax.axhline(0.5, color=pal["ink2"], lw=0.6, zorder=2)
    spread = curves.max(axis=1) - curves.min(axis=1)
    ax.fill_between(ns, curves.min(axis=1), curves.max(axis=1), color=pal["data"],
                    alpha=0.18, lw=0, zorder=1)
    ax.annotate(f"事前による幅\n（n={N_SMALL} で {spread[0]:.2f} → "
                f"n={N_FULL:,} で {spread[-1]:.2f}）",
                xy=(0.30, 0.16), xycoords="axes fraction", fontsize=5.8, color=pal["ink2"])
    ax.set_xscale("log")
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("各群の観測件数 n（対数）")
    ax.set_ylabel("$\\Pr[p_B > p_A]$")
    ax.set_title("② データが増えれば3本は重なる")

    plots.save(fig, "fig-15-9-prior-flips-conclusion.png")


def main() -> None:
    plots.setup()
    d = datasets.ab_test(n_a=N_FULL, n_b=N_FULL, p_a=P_A, lift=LIFT, seed=SEED)

    print(f"--- A/Bテスト（真値 A {d.p_a:.3f} / B {d.p_b:.3f}、seed={SEED}）---")
    print(f"    同じ1本のログを、先頭 {N_SMALL} 件で見るか {N_FULL:,} 件で見るかだけを変える\n")

    for n in (N_SMALL, N_FULL):
        k_a, k_b = int(d.a[:n].sum()), int(d.b[:n].sum())
        print(f"  n={n:>5,}   A {k_a:>4}/{n:<5} = {k_a / n:.4f}"
              f"   B {k_b:>4}/{n:<5} = {k_b / n:.4f}")
    print()

    print(f"{'事前':<24}{'説明':<26}{f'n={N_SMALL}':>10}{f'n={N_FULL:,}':>12}")
    small, full = [], []
    for name, a, b, note in PRIORS:
        p_small = prob_b_wins(d, N_SMALL, a, b)
        p_full = prob_b_wins(d, N_FULL, a, b)
        small.append(p_small)
        full.append(p_full)
        print(f"  {name:<22}{note:<24}{p_small:>10.4f}{p_full:>12.4f}")

    print(f"\n  n={N_SMALL} での開き  {max(small) - min(small):.4f}"
          f"（{min(small):.3f} 〜 {max(small):.3f}）")
    print(f"  n={N_FULL:,} での開き {max(full) - min(full):.4f}"
          f"（{min(full):.3f} 〜 {max(full):.3f}）\n")

    print(f"  閾値 {SHIP_THRESHOLD:.2f} で「Bに切り替える」と決めていたとして、n={N_SMALL} では")
    for (name, _, _, _), p in zip(PRIORS, small, strict=True):
        verdict = "切り替える" if p >= SHIP_THRESHOLD else "まだ切り替えない"
        print(f"    {name:<24}{p:.4f} → {verdict}")
    print("  データは1件も違わないのに、判断が事前の書き方で決まっている。")
    print("  一様事前と Jeffreys 事前が近く、過去実績 Beta(30,270) だけが低いのは、")
    print(f"  後者が 300 件ぶんの重さで両群を CVR {30 / 300:.2f} に引き戻しているからである。")
    print(f"  n={N_SMALL} のデータ（各群 12 件）は、その 300 件に対して 4% の発言力しかない。\n")

    print(f"  n={N_FULL:,} まで貯めると3つとも {min(full):.2f} 〜 {max(full):.2f} に収まり、")
    for (name, _, _, _), p in zip(PRIORS, full, strict=True):
        verdict = "切り替える" if p >= SHIP_THRESHOLD else "まだ切り替えない"
        print(f"    {name:<24}{p:.4f} → {verdict}")
    print("  どの事前から出発しても同じ判断になる。事前の違いはデータに洗い流された。\n")

    print("読み方は2つある。1つ目、事前を明記していないベイズA/Bの報告は読めない。")
    print("2つ目、事前で結論が動くうちは、そもそもデータが足りない。「事前を変えたら")
    print("答えが変わるか」を試すこと（感度分析）は、n が十分かを確かめる手続きでもある。")

    ns = np.unique(np.round(np.logspace(np.log10(N_SMALL), np.log10(N_FULL), 22)).astype(int))
    curves = np.array([[prob_b_wins(d, int(n), a, b) for _, a, b, _ in PRIORS] for n in ns])
    draw(d, ns, curves)


if __name__ == "__main__":
    main()
