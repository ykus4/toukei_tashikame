"""依頼4「解約しそうな人を当てたい」— 当てるのと説明するのは別の仕事。

同じ解約データに2本のモデルを当てる。予測が目的なら勾配ブースティングのほうが AUC で
勝つ。だが勾配ブースティングは「なぜ」を返さない。では説明用にロジスティック回帰の
係数を読めばよいかというと、そこにも罠がある。**共線な列を並べて入れると、係数の符号
が単変量の関連と逆を向く。**

符号が反転した変数を施策の根拠に使うと、「割引を配れば解約が減る」のような、
データにも真値にも支持されない結論が出る。予測モデルの性能が高いことは、その係数が
読めることを一切保証しない。目的関数が違うからである。

    uv run python examples/ch18/case4_churn_prediction_not_inference.py
"""

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from toukei_tashikame import glm, plots, regression

N, SEED, SPLIT_SEED = 6000, 185, 1850
N_TRAIN = 4200
NAMES = ["契約月数", "月額", "累計課金", "問合せ回数", "割引適用"]
# 真の条件付き係数（標準化スケール）。累計課金は、条件づければ効果ゼロ。
B_TRUE = np.array([-1.10, 0.30, 0.00, 0.75, 0.50])
VIF_WARN = 10.0


def standardize(v: np.ndarray) -> np.ndarray:
    return (v - v.mean()) / v.std()


def churn_data(n: int = N, seed: int = SEED) -> tuple[np.ndarray, np.ndarray]:
    """解約データを作る。実務のログに必ずある2つの癖を、わざと入れてある。

    ひとつは**派生列**。累計課金は契約月数 × 月額なので、3つを並べて入れると共線になる。
    もうひとつは**割引の割付**。長く使っている人ほど割引を受けているので、割引と解約の
    単変量の関連は、条件付きの効果と逆を向く。
    """
    rng = np.random.default_rng(seed)
    tenure = rng.gamma(2.0, 9.0, size=n)                       # 契約月数
    monthly = 3000.0 + 900.0 * rng.normal(size=n)              # 月額（円）
    total = tenure * monthly * (1 + 0.02 * rng.normal(size=n))  # 累計課金 = 派生列
    calls = rng.poisson(1.2 + 0.06 * np.maximum(0.0, 40.0 - tenure), size=n)
    # 割引は長期顧客に出ている。ランダムではない。
    disc = (rng.random(n) < 1.0 / (1.0 + np.exp(-1.2 * standardize(tenure)))).astype(float)

    X = np.column_stack([standardize(tenure), standardize(monthly), standardize(total),
                         standardize(calls), disc])
    eta = (-1.2 + X @ B_TRUE
           + 2.0 * (tenure < 6)                    # 初月の崖（線形では書けない）
           + 1.5 * X[:, 3] * (tenure < 12)         # 交互作用
           - 1.2 * np.abs(X[:, 1]))                # 月額は真ん中が最も残る
    y = (rng.random(n) < 1.0 / (1.0 + np.exp(-eta))).astype(float)
    return X, y


def draw(y_te, p_lr, p_gb, auc_lr, auc_gb, marginal, b_multi) -> None:
    fig, axes = plots.figure(1, 2, w=2.0, h=0.95)
    pal = plots.PALETTE

    ax = axes[0]
    for p, color, label, a in ((p_lr, "estimate", "ロジスティック回帰", auc_lr),
                               (p_gb, "alt", "勾配ブースティング", auc_gb)):
        fpr, tpr, _ = glm.roc(y_te, p)
        ax.plot(fpr, tpr, color=pal[color], lw=1.3, zorder=4)
        ax.annotate(f"{label}  AUC {a:.4f}",
                    xy=(0.45, 0.45 if color == "estimate" else 0.30),
                    xycoords="axes fraction", fontsize=6.2, color=pal[color])
    ax.plot([0, 1], [0, 1], color=pal["ink2"], lw=0.7, ls="--", dashes=(4, 2.2), zorder=2)
    ax.set_xlabel("偽陽性率 FPR")
    ax.set_ylabel("再現率 TPR")
    ax.set_title("① 予測の勝負。木のほうが当たる")

    ax = axes[1]
    ypos = np.arange(len(NAMES))[::-1]
    for i in range(len(NAMES)):
        y0 = ypos[i]
        flip = np.sign(marginal[i]) != np.sign(b_multi[i])
        ax.plot([marginal[i], b_multi[i]], [y0, y0],
                color=pal["reject"] if flip else pal["grid"], lw=1.2, zorder=2)
        ax.scatter([marginal[i]], [y0], s=16, color=pal["data"], zorder=4)
        ax.scatter([b_multi[i]], [y0], s=16, color=pal["estimate"], zorder=4)
        ax.scatter([B_TRUE[i]], [y0], s=22, marker="x", color=pal["truth"], zorder=5)
        if flip:
            ax.annotate("符号が反転", xy=((marginal[i] + b_multi[i]) / 2, y0),
                        xytext=(0, 5), textcoords="offset points", ha="center",
                        fontsize=5.8, color=pal["reject"])
    ax.axvline(0.0, color=pal["ink2"], lw=0.8, zorder=3)
    ax.annotate("● 単変量", xy=(0.03, 0.16), xycoords="axes fraction",
                fontsize=6.0, color=pal["data"])
    ax.annotate("● 多変量", xy=(0.03, 0.09), xycoords="axes fraction",
                fontsize=6.0, color=pal["estimate"])
    ax.annotate("✕ 真値", xy=(0.03, 0.02), xycoords="axes fraction",
                fontsize=6.0, color=pal["truth"])
    ax.set_yticks(ypos)
    ax.set_yticklabels(NAMES)
    ax.set_xlabel("係数（対数オッズ、標準化スケール）")
    ax.set_title("② 説明の勝負。係数は入れる列で向きが変わる")

    fig.tight_layout()
    plots.save(fig, "fig-18-5-prediction-vs-inference.png")


def main() -> None:
    plots.setup()
    X, y = churn_data()
    idx = np.random.default_rng(SPLIT_SEED).permutation(N)
    tr, te = idx[:N_TRAIN], idx[N_TRAIN:]
    ones = np.ones(N)

    print(f"--- 18-5 依頼4「解約しそうな人を当てたい」（n={N:,}, seed={SEED}）---\n")
    print(f"  解約率 {y.mean():.4f}   学習 {tr.size:,}件 / 検証 {te.size:,}件")

    # --- ① 予測: AUC を最大化する ---
    design = np.column_stack([ones, X])
    lr = glm.irls(design[tr], y[tr], add_const=False, names=["const", *NAMES])
    p_lr = glm.expit(design[te] @ lr.b)
    gb = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08,
                                        max_leaf_nodes=15, random_state=SEED)
    gb.fit(X[tr], y[tr])
    p_gb = gb.predict_proba(X[te])[:, 1]
    auc_lr, auc_gb = glm.auc(y[te], p_lr), glm.auc(y[te], p_gb)

    print("\n① 予測が目的なら — 検証データの AUC を比べる")
    print(f"  ロジスティック回帰   AUC {auc_lr:.4f}")
    print(f"  勾配ブースティング   AUC {auc_gb:.4f}   "
          f"（{auc_gb - auc_lr:+.4f}）")
    print("  真のモデルには「契約6ヶ月未満の崖」と交互作用が入っている。"
          "線形では書けないものを木が拾う")
    print("  ← ここまでが依頼への答え。並べ替えて上位から当たればよく、係数は要らない")

    # --- ② 説明: 係数を読もうとすると壊れる ---
    marginal = np.array([
        glm.irls(np.column_stack([ones[tr], X[tr, j]]), y[tr], add_const=False).b[1]
        for j in range(X.shape[1])
    ])
    b_multi = lr.b[1:]
    flips = [j for j in range(X.shape[1])
             if np.sign(marginal[j]) != np.sign(b_multi[j])]

    print("\n② 説明が目的なら — 同じロジスティック回帰の係数を読む")
    print("  変数        単変量     多変量      SE     真値     判定")
    for j, name in enumerate(NAMES):
        mark = "符号が反転" if j in flips else ""
        print(f"  {name:<10}{marginal[j]:>+8.3f}  {b_multi[j]:>+8.3f}  "
              f"{lr.se[j + 1]:>6.3f}  {B_TRUE[j]:>+7.2f}   {mark}")
    print(f"  符号が反転した変数 {len(flips)}個: "
          f"{'、'.join(NAMES[j] for j in flips)}")

    vifs = regression.vif(X, names=NAMES)
    print("\n  VIF（その列が他の列からどれだけ予測できるか）")
    for name, v in vifs.items():
        flag = f"  ← {VIF_WARN:.0f} 超" if v > VIF_WARN else ""
        print(f"    {name:<10}{v:>8.2f}{flag}")
    print("  累計課金 = 契約月数 × 月額 という派生列を入れたのが原因。"
          "3つのうち2つが決まれば残りはほぼ決まる")

    # --- ③ 列を1つ落とすと係数が動く ---
    keep = [0, 1, 3, 4]
    lr2 = glm.irls(np.column_stack([ones[tr], X[np.ix_(tr, keep)]]), y[tr],
                   add_const=False, names=["const", *[NAMES[j] for j in keep]])
    p_lr2 = glm.expit(np.column_stack([ones[te], X[np.ix_(te, keep)]]) @ lr2.b)
    print("\n③ 累計課金を落として当て直す")
    print("  変数        入れたとき  落としたとき   真値")
    for k, j in enumerate(keep):
        print(f"  {NAMES[j]:<10}{b_multi[j]:>+10.3f}  {lr2.b[k + 1]:>+10.3f}"
              f"  {B_TRUE[j]:>+7.2f}")
    print(f"  AUC は {auc_lr:.4f} → {glm.auc(y[te], p_lr2):.4f} でほぼ変わらない。"
          "**予測性能は落ちないのに係数だけが動く**")
    print("  ← 係数が「どの列を入れたか」で決まるなら、それは現象の性質ではない")

    # --- ④ 判定 ---
    print("\n④ 判定")
    print(f"  [{'NG  ' if flips else 'OK  '}] "
          "係数の符号が単変量と多変量で一致している")
    print(f"  [{'NG  ' if (vifs > VIF_WARN).any() else 'OK  '}] "
          f"すべての VIF が {VIF_WARN:.0f} 未満")
    print("  [NG  ] 説明変数の割付がランダム（割引は長期顧客に出ている）")
    print("\n  VERDICT: この係数を施策の根拠に使うな")
    print("  「割引を配れば解約が減る」は単変量の符号 "
          f"{marginal[4]:+.3f} を素直に読んだ結論だが、")
    print(f"  真の条件付き効果は {B_TRUE[4]:+.2f} で**逆向き**（割引を受けた人ほど解約する）。")
    print("  割引が長期顧客に偏って配られているせいで、単変量は割引ではなく契約月数を見ている")
    print("\n  予測 = 並べ替えの性能。説明 = 介入したときに何が起きるか。")
    print("  後者を言うには、第17章の道具（ランダム化・傾向スコア・DID）が要る。"
          "AUC がいくら高くても代わりにはならない")

    draw(y[te], p_lr, p_gb, auc_lr, auc_gb, marginal, b_multi)


if __name__ == "__main__":
    main()
