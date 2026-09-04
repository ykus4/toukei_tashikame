"""12店舗のCVRを、全プール・非プール・部分プールの3通りで推定する。

店舗ごとにCVRを出したい。素直な方法は2つある。

  全プール   店舗差は無いことにして、全部まとめた1つの率を全店に配る
  非プール   店舗ごとに独立に k/n を計算する

どちらも極端である。全プールは「30件しか見ていない店も4000件見た店も同じ」と言い、
非プールは「47件中3件だから 6.4%」を真に受ける。階層モデル（部分プール）はその間に
立って、**サンプルの少ない店ほど全体平均へ強く引き戻す**。引き戻す量はデータが決める。

真値を持っている合成データなので、3通りの推定を真値との二乗誤差で直接比べられる。

    uv run python examples/ch16/hierarchical_shrinkage_store_cvr.py
"""

import numpy as np
from scipy import optimize
from scipy.special import betaln, expit, logit

from toukei_tashikame import plots

SEED = 167
N_STORE = 12
P_BASE = 0.03          # 全店の平均的なCVR
TAU = 0.35             # ロジット尺度での店舗間ばらつき（真値）
N_MIN, N_MAX = 30, 4000


def make_stores(seed: int):
    """12店舗ぶんの真のCVRと観測。サンプル数は 30〜4000 で桁が違う。"""
    rng = np.random.default_rng(seed)
    p_true = expit(rng.normal(logit(P_BASE), TAU, size=N_STORE))
    trials = np.round(np.logspace(np.log10(N_MIN), np.log10(N_MAX), N_STORE)).astype(int)
    rng.shuffle(trials)
    successes = rng.binomial(trials, p_true)
    return trials, successes, p_true


def empirical_bayes(trials, successes):
    """ベータ二項の周辺尤度を最大化して事前 Beta(a, b) をデータから決める。

    階層モデルの「上の段」を、MCMC ではなく最尤で1点に決めてしまう近似
    （経験ベイズ）。事後平均は (a + k) / (a + b + n) という重み付き平均になり、
    n が小さいほど事前（＝全体平均）側の重みが勝つ。
    """
    def neg_log_marginal(theta):
        a, b = np.exp(theta)
        # 各店舗の周辺尤度 = ∫ Binom(k|n,p) Beta(p|a,b) dp（ベータ二項）
        ll = (betaln(a + successes, b + trials - successes) - betaln(a, b)).sum()
        return -ll

    start = np.log([P_BASE * 20.0, (1 - P_BASE) * 20.0])
    fit = optimize.minimize(neg_log_marginal, start, method="Nelder-Mead",
                            options={"xatol": 1e-8, "fatol": 1e-10, "maxiter": 2000})
    a, b = np.exp(fit.x)
    return float(a), float(b)


def rmse(estimate, truth) -> float:
    return float(np.sqrt(np.mean((np.asarray(estimate) - np.asarray(truth)) ** 2)))


def draw(trials, no_pool, pooled, partial, p_true, errs) -> None:
    fig, axes = plots.figure(1, 2, w=2.0, h=1.0, gridspec_kw={"width_ratios": [1.7, 1.0]})
    pal = plots.PALETTE

    ax = axes[0]
    for n, lo, hi in zip(trials, no_pool, partial, strict=True):
        ax.annotate("", xy=(n, hi), xytext=(n, lo),
                    arrowprops={"arrowstyle": "-|>", "color": pal["ink2"],
                                "lw": 0.6, "shrinkA": 1.0, "shrinkB": 1.0})
    ax.scatter(trials, no_pool, s=14, color=pal["data"], zorder=4)
    ax.scatter(trials, partial, s=14, color=pal["posterior"], zorder=5)
    ax.scatter(trials, p_true, s=16, marker="x", color=pal["truth"], zorder=6, lw=0.9)
    ax.axhline(pooled, color=pal["reject"], lw=1.0, ls="--", dashes=(4, 2.2), zorder=3)
    ax.annotate(f"全プール {pooled:.4f}", xy=(N_MAX * 0.9, pooled), ha="right",
                va="bottom", fontsize=6.0, color=pal["reject"])
    ax.annotate("灰=非プール（k/n）  青=部分プール  ×=真値",
                xy=(0.02, 0.97), xycoords="axes fraction", va="top", fontsize=6.0,
                color=pal["ink2"])
    top = float(max(no_pool.max(), p_true.max()))
    bottom = float(min(no_pool.min(), p_true.min()))
    ax.set_ylim(bottom - 0.003, top + 0.010)   # 上に凡例ぶんの余白を空ける
    ax.set_xscale("log")
    ax.set_xlabel("店舗のサンプル数（対数目盛）")
    ax.set_ylabel("CVR")
    ax.set_title("① 少ない店ほど強く引き戻される")

    ax = axes[1]
    names = ["非プール", "全プール", "部分プール"]
    vals = [errs["no_pool"], errs["pooled"], errs["partial"]]
    ax.bar(names, vals, color=[pal["data"], pal["reject"], pal["posterior"]], width=0.55)
    for i, v in enumerate(vals):
        ax.annotate(f"{v:.4f}", xy=(i, v), ha="center", va="bottom", fontsize=6.0,
                    color=pal["ink2"])
    ax.set_ylim(0, max(vals) * 1.25)
    ax.set_ylabel("真値との RMSE")
    ax.set_title("② 部分プールが一番当たる")

    plots.save(fig, "fig-16-7-shrinkage.png")


def main() -> None:
    plots.setup()
    trials, successes, p_true = make_stores(SEED)

    no_pool = successes / trials                       # 非プール
    pooled = float(successes.sum() / trials.sum())     # 全プール
    a, b = empirical_bayes(trials, successes)
    partial = (a + successes) / (a + b + trials)       # 部分プール（事後平均）
    weight = trials / (a + b + trials)                 # データ側の重み

    errs = {"no_pool": rmse(no_pool, p_true), "pooled": rmse(pooled, p_true),
            "partial": rmse(partial, p_true)}

    print(f"--- 12店舗のCVR（真値の中心 {P_BASE:.1%}、seed={SEED}）---")
    print(f"  データから決まった事前: Beta({a:.2f}, {b:.2f})"
          f" → 事前平均 {a / (a + b):.4f}、擬似サンプル数 {a + b:.1f} 件ぶん\n")

    order = np.argsort(trials)
    print(f"{'店舗':>6}{'件数':>8}{'CV数':>7}{'非プール':>11}{'全プール':>11}"
          f"{'部分プール':>12}{'データ側の重み':>16}{'真値':>10}")
    for i in order:
        print(f"{i:>6}{trials[i]:>8,}{successes[i]:>7}{no_pool[i]:>11.4f}"
              f"{pooled:>11.4f}{partial[i]:>12.4f}{weight[i]:>16.3f}{p_true[i]:>10.4f}")

    print(f"\n  真値との RMSE:  非プール {errs['no_pool']:.4f} / "
          f"全プール {errs['pooled']:.4f} / 部分プール {errs['partial']:.4f}")
    print(f"  部分プールは非プールの {errs['partial'] / errs['no_pool']:.2f} 倍、"
          f"全プールの {errs['partial'] / errs['pooled']:.2f} 倍の誤差で済んでいる。\n")

    small = int(order[0])
    big = int(order[-1])
    print(f"  一番小さい店舗{small}（{trials[small]} 件中 {successes[small]} 件）:")
    print(f"    非プール {no_pool[small]:.4f} → 部分プール {partial[small]:.4f}"
          f"（真値 {p_true[small]:.4f}）。データ側の重みは {weight[small]:.2f} しかない。")
    print(f"  一番大きい店舗{big}（{trials[big]:,} 件中 {successes[big]} 件）:")
    print(f"    非プール {no_pool[big]:.4f} → 部分プール {partial[big]:.4f}"
          f"（真値 {p_true[big]:.4f}）。重み {weight[big]:.2f} で、ほぼ動かない。")
    moved = int(np.argmax(np.abs(partial - no_pool)))
    print(f"  一番大きく動いた店舗{moved}（{trials[moved]} 件中 {successes[moved]} 件）:")
    print(f"    非プール {no_pool[moved]:.4f} → 部分プール {partial[moved]:.4f}"
          f"（真値 {p_true[moved]:.4f}）。少ない件数で出た高い率は、"
          "そのまま信じない。\n")

    print("  引き戻す量を人が決めていないことが大事である。全体のばらつきが小さければ")
    print("  強く引き戻し、店舗差が本当に大きければ引き戻さない。その判断を、"
          "上の段の事前")
    print("  Beta(a, b) をデータから推す形で自動化したのが階層モデルである。")
    print("  ここでは経験ベイズ（上の段を1点に決める近似）で書いた。PyMC で上の段にも")
    print("  事前を置いて丸ごと事後を出すのが 16-8 の階層ベータ二項モデルになる。")

    draw(trials, no_pool, pooled, partial, p_true, errs)


if __name__ == "__main__":
    main()
