"""効果がゼロのログから有意差を作る4つのレシピを、偽陽性率で値踏みする。

p ハッキングは捏造ではない。どのレシピも「やってはいけない計算」を1つも含まない。
やっているのは**複数の分析を試して、都合のよい1つを報告する**ことだけである。それが
名目5%の第一種の誤りを何倍にするかを、真の差がゼロだと分かっているログの上で数える。

素直に1回だけ検定する手続きと、①途中で覗く ②外れ値を1つ消す ③サブグループを切る
④指標を3つ試す、そして全部盛りを、同じデータ生成から 10,000 回ずつ回して比べる。

    uv run python examples/ch09/p_hacking_four_recipes.py
"""

import numpy as np
from scipy import special

from toukei_tashikame import plots, sim

N = 50                              # 群あたりの人数（最終的な到達点）
LOOKS = (10, 20, 30, 40, 50)        # 途中で覗くタイミング
ALPHA = 0.05
TRIALS = 10_000
SEED = 91

RECIPES = [
    "素直に1回だけ検定する",
    "① 途中で5回覗く",
    "② 外れ値を1つ消してみる",
    "③ サブグループを切る",
    "④ 指標を3つ試す",
    "全部盛り（①〜④の最小p）",
]


def welch_p(a: np.ndarray, b: np.ndarray) -> float:
    """Welch の t 検定の両側 p 値。手で書く。

    ここだけ scipy の ``ttest_ind`` を使わないのは速度の都合で、1試行あたり十数回の
    検定を 10,000 回ぶん回すと、関数呼び出しの間接費のほうが計算より高くつくため。
    式は第8章の手書き版とまったく同じものである。
    """
    n1, n2 = a.size, b.size
    if n1 < 3 or n2 < 3:
        return 1.0
    s1, s2 = a.var(ddof=1) / n1, b.var(ddof=1) / n2
    if s1 + s2 <= 0.0:
        return 1.0
    t = (a.mean() - b.mean()) / np.sqrt(s1 + s2)
    df = (s1 + s2) ** 2 / (s1**2 / (n1 - 1) + s2**2 / (n2 - 1))
    return float(2 * special.stdtr(df, -abs(t)))


def drop_one_outlier(a: np.ndarray, b: np.ndarray) -> float:
    """群内で最も外れた1点を落として検定し直し、小さいほうの p を採る。"""
    za = np.abs(a - a.mean()) / a.std(ddof=1)
    zb = np.abs(b - b.mean()) / b.std(ddof=1)
    if za.max() >= zb.max():
        return welch_p(np.delete(a, za.argmax()), b)
    return welch_p(a, np.delete(b, zb.argmax()))


def one_trial(rng) -> tuple[float, ...]:
    """真の差がゼロのログを1本作り、6つの手続きそれぞれの p 値を返す。"""
    # 指標3本。どれも A と B で分布は同じ（真の効果はゼロ）。
    a = rng.normal(0.0, 1.0, size=(3, N))
    b = rng.normal(0.0, 1.0, size=(3, N))
    # 事後に切りたくなるサブグループ（新規/既存など）。結果とは何の関係もない。
    seg_a = rng.integers(0, 2, size=N)
    seg_b = rng.integers(0, 2, size=N)

    honest = welch_p(a[0], b[0])

    # ① 途中で覗く。有意になった時点の p を採る（＝覗いた中の最小 p と同じ判定）。
    peeking = min(welch_p(a[0, :k], b[0, :k]) for k in LOOKS)

    # ② 外れ値を1つ消す。消すか消さないかを結果を見てから決める。
    outlier = min(honest, drop_one_outlier(a[0], b[0]))

    # ③ サブグループを切る。全体と2つのセグメントの中から一番良いものを採る。
    subgroup = min(
        honest,
        welch_p(a[0][seg_a == 0], b[0][seg_b == 0]),
        welch_p(a[0][seg_a == 1], b[0][seg_b == 1]),
    )

    # ④ 指標を3つ試す。
    metrics = min(welch_p(a[i], b[i]) for i in range(3))

    return honest, peeking, outlier, subgroup, metrics, min(
        peeking, outlier, subgroup, metrics
    )


def draw(rates: np.ndarray, ses: np.ndarray) -> None:
    fig, ax = plots.figure(h=1.25)
    pos = np.arange(len(RECIPES))[::-1]
    colors = [plots.PALETTE["data"]] + [plots.PALETTE["reject"]] * (len(RECIPES) - 1)
    ax.barh(pos, rates, height=0.62, color=colors, lw=0)
    ax.errorbar(rates, pos, xerr=1.96 * ses, fmt="none",
                ecolor=plots.PALETTE["ink2"], elinewidth=0.7, capsize=1.6)
    for y, r, se in zip(pos, rates, ses, strict=True):
        ax.annotate(f"{r:.4f}", xy=(r + 1.96 * se, y), xytext=(5, 0),
                    textcoords="offset points", va="center", fontsize=6.0,
                    color=plots.PALETTE["ink"])
    plots.mark_truth(ax, ALPHA, "名目 α = 0.05")
    ax.set_yticks(pos, RECIPES, fontsize=6.2)
    ax.set_xlabel("偽陽性率（真の差はゼロ）")
    ax.set_xlim(0, max(rates) * 1.22)
    ax.set_title(f"p ハッキングの4レシピ（各 {TRIALS:,} 回, n={N}/群）")
    plots.save(fig, "fig-9-1-p-hacking-recipes.png")


def main() -> None:
    plots.setup()
    with sim.Timer("9-1 pハッキング"):
        p = sim.repeat(one_trial, trials=TRIALS, seed=SEED, progress=False)

    rates = (p < ALPHA).mean(axis=0)
    ses = np.sqrt(rates * (1 - rates) / TRIALS)

    print(f"真の差はゼロ。n={N}/群、名目 α={ALPHA}、各手続き {TRIALS:,} 回\n")
    print(f"{'手続き':<26}{'偽陽性率':>10}{'±95%':>9}{'名目の何倍':>10}")
    for name, r, se in zip(RECIPES, rates, ses, strict=True):
        print(f"{name:<26}{r:>10.4f}{1.96 * se:>9.4f}{r / ALPHA:>9.1f}倍")

    print("\nどのレシピも「間違った計算」は1つもしていない。試した分析の数だけを隠している。")
    draw(rates, ses)


if __name__ == "__main__":
    main()
