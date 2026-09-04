"""「とりあえず全部入れる」が効果を作り出す — 合流点を調整したときの偽陽性。

交絡は入れなければならない。だから「関係ありそうな変数は全部入れる」が安全側だと
思われがちだが、これは誤りである。処置と結果の**両方が原因になっている**変数
（合流点, collider）を入れると、無かったはずの関連が生まれる。

ここでは処置の真の効果を 0 にしてある。素直に $Y \\sim T$ と回帰すれば棄却率は
名目どおり 5% で収まる。そこに合流点 $C$ を1つ足すだけで、棄却率は跳ね上がり、
推定値は 0 から離れたところに固まる。**変数を足す判断は、データではなく因果の図で
決めるしかない。**

    uv run python examples/ch17/collider_bias_adjusting_too_much.py
"""

import numpy as np
from scipy import stats

from toukei_tashikame import plots, sim

N, TRUE_EFFECT, SEED = 2000, 0.0, 176
A_T, A_Y = 1.0, 1.0     # T → C と Y → C の強さ
TRIALS, ALPHA = 10_000, 0.05


def make_data(rng, n: int = N):
    """T → C ← Y。C は処置と結果の下流にあり、両方の子である。"""
    t = (rng.random(n) < 0.5).astype(float)
    y = TRUE_EFFECT * t + rng.normal(0.0, 1.0, size=n)
    c = A_T * t + A_Y * y + rng.normal(0.0, 1.0, size=n)   # 合流点（例: 問い合わせ有無）
    return t, y, c


def slope_and_p(X, y):
    """列 1 の係数と両側 p 値。正規方程式を直に解く（毎回 OLS を呼ぶと遅い）。"""
    xtx = X.T @ X
    b = np.linalg.solve(xtx, X.T @ y)
    resid = y - X @ b
    n, k = X.shape
    sigma2 = float(resid @ resid) / (n - k)
    se = np.sqrt(sigma2 * np.linalg.inv(xtx)[1, 1])
    return float(b[1]), float(2 * stats.t.sf(abs(b[1] / se), df=n - k))


def one_trial(rng):
    t, y, c = make_data(rng)
    ones = np.ones(t.size)
    b0, p0 = slope_and_p(np.column_stack([ones, t]), y)          # 調整なし
    b1, p1 = slope_and_p(np.column_stack([ones, t, c]), y)       # 合流点を調整
    return b0, p0, b1, p1


def draw(out) -> None:
    fig, axes = plots.figure(1, 2, w=2.0)
    pal = plots.PALETTE

    # 左: DAG。C は T と Y の子であって、親ではない。
    ax = axes[0]
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    nodes = {"T（処置）": (0.15, 0.75), "Y（結果）": (0.85, 0.75), "C（合流点）": (0.50, 0.25)}
    for name, (px, py) in nodes.items():
        ax.annotate(name, xy=(px, py), ha="center", va="center", fontsize=7.0,
                    color=pal["ink"],
                    bbox={"boxstyle": "round,pad=0.35", "facecolor": "white",
                          "edgecolor": pal["ink2"], "linewidth": 0.8})
    for src, dst, color in (("T（処置）", "C（合流点）", pal["reject"]),
                            ("Y（結果）", "C（合流点）", pal["reject"])):
        ax.annotate("", xy=nodes[dst], xytext=nodes[src],
                    arrowprops={"arrowstyle": "-|>", "color": color, "lw": 1.1,
                                "shrinkA": 22, "shrinkB": 22})
    ax.annotate("", xy=nodes["Y（結果）"], xytext=nodes["T（処置）"],
                arrowprops={"arrowstyle": "-|>", "color": pal["grid"], "lw": 1.1,
                            "shrinkA": 26, "shrinkB": 26, "linestyle": "--"})
    ax.annotate("真の効果 0（矢印は無い）", xy=(0.5, 0.80), ha="center", fontsize=6.2,
                color=pal["ink2"])
    ax.annotate("C で条件づけると\nT と Y がつながる", xy=(0.5, 0.06), ha="center",
                fontsize=6.2, color=pal["reject"])

    # 右: 推定値の分布。調整なしは 0 の周り、合流点を入れると 0 から離れる。
    ax = axes[1]
    bins = np.linspace(min(out[:, 2].min(), out[:, 0].min()),
                       max(out[:, 2].max(), out[:, 0].max()), 70)
    ax.hist(out[:, 0], bins=bins, color=pal["estimate"], alpha=0.55, lw=0)
    ax.hist(out[:, 2], bins=bins, color=pal["reject"], alpha=0.55, lw=0)
    plots.mark_truth(ax, TRUE_EFFECT, "真の効果 = 0")
    ax.annotate("調整なし", xy=(out[:, 0].mean(), 0.70), xycoords=("data", "axes fraction"),
                ha="center", fontsize=6.2, color=pal["estimate"])
    ax.annotate("C を調整", xy=(out[:, 2].mean(), 0.70), xycoords=("data", "axes fraction"),
                ha="center", fontsize=6.2, color=pal["reject"])
    ax.set_xlabel(f"処置の係数（n={N:,} を {TRIALS:,} 回）")
    ax.set_ylabel("回数")
    fig.tight_layout()
    plots.save(fig, "fig-17-6-collider-bias.png")


def main() -> None:
    plots.setup()
    with sim.Timer(f"{TRIALS:,} 回ぶんの当てはめ") as timer:
        out = sim.repeat(one_trial, trials=TRIALS, seed=SEED, progress=False)

    print(f"--- T → C ← Y の合流点を作り、{TRIALS:,} 回ずつ回帰する"
          f"（n={N:,}, 真の効果 {TRUE_EFFECT:g}, seed={SEED}）---\n")
    print(f"{'モデル':<26}{'平均推定':>10}{'SD':>8}{'棄却率(α=0.05)':>16}")
    for label, cols in (("Y ~ T（調整なし）", (0, 1)), ("Y ~ T + C（合流点を調整）", (2, 3))):
        est, p = out[:, cols[0]], out[:, cols[1]]
        rate = float((p < ALPHA).mean())
        se = np.sqrt(rate * (1 - rate) / TRIALS)
        print(f"{label:<26}{est.mean():>10.3f}{est.std(ddof=1):>8.3f}"
              f"{rate:>13.4f} ± {1.96 * se:.4f}")

    print(f"\n  調整なしの棄却率は名目の {ALPHA:.0%} と一致する。真の効果が 0 なので、これが正しい姿。")
    print(f"  C を入れると棄却率が {float((out[:, 3] < ALPHA).mean()):.4f} まで上がる。"
          "検定が壊れたのではなく、")
    print("  「C を固定した中での T と Y の関連」を正しく測ってしまっている。")
    print(f"  推定値も 0 ではなく {out[:, 2].mean():.3f} の周りに固まる。"
          "n を増やしてもここへ収束する。")

    print("\n  なぜつながるか: C ≈ T + Y なので、C を固定すると T が大きい人ほど Y は小さい。")
    print("  例えるなら「問い合わせをした人」だけを見ると、機能を使った人ほど満足度が低く見える。")
    print("  使っていないのに問い合わせるのは、よほど困った人だけだからである。")

    rng = np.random.default_rng(SEED)
    t, y, c = make_data(rng)
    hi = c > np.median(c)
    print(f"\n  1回ぶんのデータで、C の高い層だけを見た相関 = "
          f"{np.corrcoef(t[hi], y[hi])[0, 1]:+.3f}")
    print(f"  低い層だけ = {np.corrcoef(t[~hi], y[~hi])[0, 1]:+.3f}、"
          f"層に分けなければ = {np.corrcoef(t, y)[0, 1]:+.3f}")
    print("  層の中でだけ負になる。これが「調整」が作り出した関連の正体である。")

    print("\n  実務での見分け方は1つしかない: その変数は処置より前に決まっていたか。")
    print("  処置のあとに動く変数（利用回数・問い合わせ・解約など）は、入れる前に図を書く。")
    print(f"\n  （回帰 {2 * TRIALS:,} 本ぶんで {timer.elapsed:.2f} 秒）")

    draw(out)


if __name__ == "__main__":
    main()
