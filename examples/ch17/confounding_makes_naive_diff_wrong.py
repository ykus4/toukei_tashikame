"""交絡があると、素朴な群間差は因果効果ではない。真値を知っているデータで確かめる。

`datasets.observational` は真の ATE を返り値に持っている。実データなら永久に分からない
「答え」が手元にあるので、素朴な群間差がどちらへ何倍ずれるかを直接測れる。

同じデータに共変量 $Z$ を入れた回帰を当てると真値に戻る。ただしそれは
**交絡変数が観測できている**からで、実務でこの前提が成り立っているかどうかは
データからは分からない。最後に交絡の強さを 0 から振って、バイアスが強さとともに
どう伸びるかを見る。強さ 0 の点だけが、調整しなくてよい世界である。

    uv run python examples/ch17/confounding_makes_naive_diff_wrong.py
"""

import numpy as np

from toukei_tashikame import causal, datasets, plots, regression

N, ATE, CONFOUNDING, SEED = 2000, 1.5, 1.5, 175
STRENGTHS = np.linspace(0.0, 3.0, 13)   # 交絡の強さを振る（本編の設定は 1.5）


def adjusted(d) -> regression.OlsResult:
    """処置と交絡変数を同時に入れた回帰。処置の係数が調整後の推定値になる。"""
    X = np.column_stack([np.ones(d.x.size), d.x, d.z])
    return regression.ols(X, d.y, add_const=False, names=["const", "処置 T", "交絡 Z"])


def bias_curve():
    """交絡の強さごとに、素朴差と調整後がどれだけずれるかを測る。"""
    rows = []
    for c in STRENGTHS:
        d = datasets.observational(n=N, ate=ATE, confounding=float(c), seed=SEED)
        rows.append((c, d.naive_diff - ATE, adjusted(d).b[1] - ATE))
    return np.array(rows)


def draw(curve) -> None:
    fig, ax = plots.figure(w=1.4)
    pal = plots.PALETTE
    ax.plot(curve[:, 0], curve[:, 1], color=pal["reject"], lw=1.4, marker="o", ms=3,
            zorder=4)
    ax.plot(curve[:, 0], curve[:, 2], color=pal["estimate"], lw=1.4, marker="o", ms=3,
            zorder=4)
    plots.mark_truth(ax, 0.0, "バイアス 0（真値と一致）", axis="y")
    ax.axvline(CONFOUNDING, color=pal["ink2"], lw=0.8, ls="--", dashes=(4, 2.0), zorder=2)
    ax.annotate("本編の設定 1.5", xy=(CONFOUNDING, 0.06), xycoords=("data", "axes fraction"),
                fontsize=6.0, color=pal["ink2"], ha="left",
                xytext=(3, 0), textcoords="offset points")
    ax.annotate("素朴な群間差", xy=(curve[-1, 0], curve[-1, 1]), fontsize=6.2,
                color=pal["reject"], ha="right", va="bottom",
                xytext=(-2, 3), textcoords="offset points")
    mid = len(curve) // 3
    ax.annotate("Z で調整した回帰", xy=(curve[mid, 0], curve[mid, 2]), fontsize=6.2,
                color=pal["estimate"], ha="center", va="top",
                xytext=(0, -5), textcoords="offset points")
    ax.set_xlabel("交絡の強さ（Z → T と Z → Y の係数）")
    ax.set_ylabel("真の ATE からのずれ")
    fig.tight_layout()
    plots.save(fig, "fig-17-5-confounding-bias-curve.png")


def main() -> None:
    plots.setup()
    d = datasets.observational(n=N, ate=ATE, confounding=CONFOUNDING, seed=SEED)

    naive = causal.naive_diff(d.y, d.x)
    adj = adjusted(d)
    lo, hi = adj.conf_int()[1]

    print(f"--- 観察データ n={N:,}、真の ATE = {ATE}、交絡の強さ {CONFOUNDING}"
          f"（seed={SEED}）---\n")
    print(f"  処置群 {int(d.x.sum()):,}人 / 対照群 {int((1 - d.x).sum()):,}人")
    print(f"  交絡変数 Z の群平均: 処置群 {d.z[d.x == 1].mean():+.3f} / "
          f"対照群 {d.z[d.x == 0].mean():+.3f}  ← ここが揃っていない")

    print(f"\n{'推定量':<24}{'推定値':>9}{'真値との比':>12}{'ずれ':>9}")
    print(f"{'素朴な群間差':<24}{naive.estimate:>9.3f}{naive.estimate / ATE:>11.2f}倍"
          f"{naive.estimate - ATE:>+9.3f}")
    print(f"{'Z で調整した回帰':<24}{adj.b[1]:>9.3f}{adj.b[1] / ATE:>11.2f}倍"
          f"{adj.b[1] - ATE:>+9.3f}")
    print(f"{'真の ATE（合成データだけが知る）':<24}{ATE:>9.3f}{1.0:>11.2f}倍{0.0:>+9.3f}")

    print(f"\n  調整後の 95%CI [{lo:.3f}, {hi:.3f}] は真値 {ATE} を含む。"
          f"素朴差の 95%CI [{naive.ci[0]:.3f}, {naive.ci[1]:.3f}] は含まない。")
    print(f"  素朴差の 95%CI は幅 {naive.ci[1] - naive.ci[0]:.3f} と狭い。"
          "狭いことは正しさの証拠にならない。")

    print("\n  素朴差が上振れする内訳:")
    print(f"    処置の効果          {ATE:+.3f}")
    print(f"    Z 経由の見かけの差   {naive.estimate - ATE:+.3f}"
          "   ← 処置群のほうが元から Z が高い")
    print(f"    合計                {naive.estimate:+.3f}")

    print(f"\n{naive!s}\n")

    curve = bias_curve()
    print(f"--- 交絡の強さを 0〜3 で振る（各点で n={N:,} を1回引く）---\n")
    print(f"{'強さ':>6}{'素朴差のずれ':>14}{'調整後のずれ':>14}")
    for c, b_naive, b_adj in curve:
        print(f"{c:>6.2f}{b_naive:>+14.3f}{b_adj:>+14.3f}")
    print("\n  強さ 0 の行だけが、調整しなくても素朴差が使える世界（＝ランダム化）。")
    print("  そこから外れた瞬間にずれが立ち上がり、強さにほぼ比例して伸びる。")
    print("  調整後の列はどの強さでも 0 の近くに留まる——Z が観測できているから。")
    print("  観測できない交絡が1つでもあれば、この列も素朴差と同じように伸びる。")

    draw(curve)


if __name__ == "__main__":
    main()
