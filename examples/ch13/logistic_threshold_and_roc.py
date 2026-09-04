"""0.5 は既定であって正解ではない — 閾値を動かして混同行列を見る。

ロジスティック回帰が返すのは確率であって、0/1 の判定ではない。判定にするには
閾値がいる。ここを 0.5 に固定したまま「精度が低い」と言っている場面はとても多い。
閾値を 0.05 刻みで動かすと、適合率と再現率が正反対に動き、F1 を最大にする点が
0.5 とは限らないことがすぐ見える。

そのうえで、閾値に依存しない指標として AUC を台形則で手計算し、
``sklearn.metrics.roc_auc_score`` と突き合わせる。AUC は「ランダムに選んだ陽性が、
ランダムに選んだ陰性より高いスコアを得る確率」でもあるので、その数え上げとも
一致することを確かめる。

    uv run python examples/ch13/logistic_threshold_and_roc.py
"""

import numpy as np
from sklearn.metrics import roc_auc_score

from toukei_tashikame import datasets, glm, plots

N, B_TRUE, SEED = 1_000, (-0.7, 1.2), 135


def scores(y, p, threshold):
    """適合率・再現率・F1 を混同行列から組み立てる。"""
    pred = p >= threshold
    tp = int(((y == 1) & pred).sum())
    fp = int(((y == 0) & pred).sum())
    fn = int(((y == 1) & ~pred).sum())
    tn = int(((y == 0) & ~pred).sum())
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if tp else 0.0
    return tp, fp, fn, tn, prec, rec, f1


def main() -> None:
    plots.setup()
    X, y, b_true = datasets.logistic_data(N, b=B_TRUE, seed=SEED)
    fit = glm.irls(X, y, add_const=False, names=["const", "x"])
    p = fit.predict()

    print(f"--- 13-5 閾値を動かす（n={N}, seed={SEED}, 陽性率 {y.mean():.4f}）---")
    print(f"  当てはめた係数 切片 {fit.b[0]:+.4f} / 傾き {fit.b[1]:+.4f}"
          f"（真値 {b_true[0]:+.2f} / {b_true[1]:+.2f}）")
    print("  閾値    TP   FP   FN   TN    適合率   再現率     F1")
    thresholds = np.round(np.arange(0.05, 1.00, 0.05), 2)
    table = []
    for t in thresholds:
        tp, fp, fn, tn, prec, rec, f1 = scores(y, p, t)
        table.append((t, prec, rec, f1))
        print(f"  {t:.2f}  {tp:4d} {fp:4d} {fn:4d} {tn:4d}   {prec:.4f}   {rec:.4f}"
              f"   {f1:.4f}")
    arr = np.array(table)
    best = int(np.nanargmax(arr[:, 3]))
    t50 = int(np.argmin(np.abs(thresholds - 0.50)))
    print(f"\n  閾値 0.50   適合率 {arr[t50, 1]:.4f} / 再現率 {arr[t50, 2]:.4f}"
          f" / F1 {arr[t50, 3]:.4f}")
    print(f"  F1 最大は閾値 {arr[best, 0]:.2f}   適合率 {arr[best, 1]:.4f}"
          f" / 再現率 {arr[best, 2]:.4f} / F1 {arr[best, 3]:.4f}")
    print("  ← 0.5 のままだと F1 で "
          f"{arr[best, 3] - arr[t50, 3]:+.4f}、再現率で {arr[best, 2] - arr[t50, 2]:+.4f} 損している")
    print("\n  閾値 0.50 の混同行列")
    print(glm.confusion(y, p, 0.50).to_string())

    # --- AUC を台形則で手計算する ---
    fpr, tpr, _ = glm.roc(y, p)
    auc_trapz = float(np.sum((fpr[1:] - fpr[:-1]) * (tpr[1:] + tpr[:-1]) / 2))
    pos, neg = p[y == 1], p[y == 0]
    auc_pairs = float(np.mean(pos[:, None] > neg[None, :])
                      + 0.5 * np.mean(pos[:, None] == neg[None, :]))
    print("\n--- AUC を3通りで出す（閾値に依存しない指標）---")
    print(f"  台形則を手で書く          {auc_trapz:.6f}")
    print(f"  glm.auc（np.trapezoid）   {glm.auc(y, p):.6f}")
    print(f"  sklearn.roc_auc_score     {roc_auc_score(y, p):.6f}")
    print(f"  陽性 > 陰性 の対の割合    {auc_pairs:.6f}"
          f"   ← {pos.size:,} × {neg.size:,} = {pos.size * neg.size:,} 対を数えただけ")
    print(f"  手計算 vs sklearn の差    {abs(auc_trapz - roc_auc_score(y, p)):.3e}")

    # --- 図 ---
    fig, axes = plots.figure(1, 2, w=1.8, h=1.0)
    ax = axes[0]
    ax.plot(fpr, tpr, color=plots.PALETTE["estimate"], lw=1.3, zorder=4)
    ax.fill_between(fpr, tpr, color=plots.PALETTE["interval"], alpha=0.20, lw=0, zorder=1)
    ax.plot([0, 1], [0, 1], color=plots.PALETTE["ink2"], lw=0.8, ls=":", zorder=3)
    for t, marker in ((0.50, "o"), (float(arr[best, 0]), "s")):
        _, _, _, _, _, rec, _ = scores(y, p, t)
        far = float(((y == 0) & (p >= t)).sum() / (y == 0).sum())
        ax.scatter([far], [rec], marker=marker, s=22, color=plots.PALETTE["reject"],
                   zorder=6)
        ax.annotate(f"閾値 {t:.2f}", xy=(far, rec), xytext=(5, -3),
                    textcoords="offset points", fontsize=6.0,
                    color=plots.PALETTE["reject"])
    ax.annotate(f"AUC = {auc_trapz:.4f}", xy=(0.55, 0.30), fontsize=6.6,
                color=plots.PALETTE["estimate"])
    ax.annotate("当てずっぽう", xy=(0.72, 0.72), xytext=(0, -9),
                textcoords="offset points", fontsize=6.0, color=plots.PALETTE["ink2"])
    ax.set_xlabel("偽陽性率 FPR")
    ax.set_ylabel("再現率 TPR")
    ax.set_title("ROC 曲線（閾値を全部通った軌跡）")

    ax = axes[1]
    ax.plot(arr[:, 0], arr[:, 1], color=plots.PALETTE["estimate"], lw=1.2, zorder=4)
    ax.plot(arr[:, 0], arr[:, 2], color=plots.PALETTE["alt"], lw=1.2, zorder=4)
    ax.plot(arr[:, 0], arr[:, 3], color=plots.PALETTE["reject"], lw=1.4, zorder=5)
    ax.axvline(arr[best, 0], color=plots.PALETTE["reject"], lw=0.9, ls="--",
               dashes=(4, 2.2), zorder=3)
    ax.annotate("適合率", xy=(arr[-3, 0], arr[-3, 1]), xytext=(-2, 4),
                textcoords="offset points", fontsize=6.0, ha="right",
                color=plots.PALETTE["estimate"])
    ax.annotate("再現率", xy=(arr[2, 0], arr[2, 2]), xytext=(2, -8),
                textcoords="offset points", fontsize=6.0, color=plots.PALETTE["alt"])
    ax.annotate(f"F1 最大 {arr[best, 0]:.2f}", xy=(arr[best, 0], arr[best, 3]),
                xytext=(4, 4), textcoords="offset points", fontsize=6.0,
                color=plots.PALETTE["reject"])
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("閾値")
    ax.set_ylabel("値")
    ax.set_title("閾値を動かすと、適合率と再現率は逆に動く")
    fig.tight_layout()
    plots.save(fig, "fig-13-5-roc-and-threshold.png")


if __name__ == "__main__":
    main()
