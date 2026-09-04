"""ロジスティック回帰が答えを返せない2つの場面 — 完全分離と、まれな事象。

**完全分離**は、ある変数の値で 0 と 1 が完全に分かれてしまう状態を言う。このとき尤度は
係数を大きくするほど増え続け、上限に達しない。最尤推定量が存在しないので、IRLS は
止まらず、係数はいくらでも大きくなる。困るのは、それが「エラー」ではなく
「巨大な係数と、それ以上に巨大な標準誤差」として返ってくることで、Wald 検定の p 値は
1 に近づく——**係数は無限大なのに有意でない**、という読めない出力になる。

**まれな事象**は分離の手前の状態である。事象が数個しかないと、最尤推定は有限だが偏る。
偏りは主に切片に出て、確率を系統的に低く見積もる。どちらも Firth の罰則付き尤度が
効く。罰則項 $\\frac{1}{2}\\log|X^\\top W X|$ を足すだけで推定量は必ず有限になり、
$O(1/n)$ の偏りが消える。Firth は本書の道具にないので、ここで手で書く（15行）。

    uv run python examples/ch13/complete_separation_and_rare_event_bias.py
"""

import warnings

import numpy as np
import statsmodels.api as sm

from toukei_tashikame import glm, plots, sim

N_SEP, SEED_SEP = 60, 138
N_RARE, B_RARE, SEED_RARE, TRIALS = 500, (-5.0, 1.0), 138, 3_000
DIVERGED = 10.0   # |係数| がこれを超えたら、分離して発散したものとして数える


def firth_logit(X, y, max_iter: int = 100, tol: float = 1e-10):
    """Firth の罰則付きロジスティック回帰。IRLS のスコアに補正項を足すだけ。

    罰則 $\\frac{1}{2}\\log|X^\\top W X|$（Jeffreys 事前分布）の微分は
    $h_i(1/2-\\mu_i)$ という形になる。$h_i$ はハット行列の対角、つまり観測 i の
    てこ比で、**てこの大きい点の y を 0.5 のほうへ少しだけ引き戻す**罰則である。
    データを完全に説明しきる係数が罰されるので、推定量は必ず有限になる。
    """
    b = np.zeros(X.shape[1])
    for _ in range(max_iter):
        mu = glm.expit(X @ b)
        w = np.clip(mu * (1.0 - mu), 1e-10, None)
        inv = np.linalg.inv(X.T @ (X * w[:, None]))
        h = np.einsum("ij,jk,ik->i", X, inv, X) * w      # ハット行列の対角
        b_new = b + inv @ (X.T @ (y - mu + h * (0.5 - mu)))
        if np.max(np.abs(b_new - b)) < tol:
            return b_new
        b = b_new
    return b


def separation_demo():
    """完全分離するデータを作って、係数が止まらない様子を出す。"""
    rng = np.random.default_rng(SEED_SEP)
    x = rng.normal(0.0, 1.0, size=N_SEP)
    y = (x > 0.0).astype(float)      # x の符号で 0/1 が完全に分かれる
    X = np.column_stack([np.ones(N_SEP), x])

    gap_lo, gap_hi = x[y == 0].max(), x[y == 1].min()
    print(f"--- 13-8 完全分離（n={N_SEP}, seed={SEED_SEP}）---")
    print(f"  y=1 は {int(y.sum())} 件、y=0 は {int(N_SEP - y.sum())} 件")
    print(f"  y=0 側の x の最大 {gap_lo:+.4f} < y=1 側の x の最小 {gap_hi:+.4f}"
          f"   ← 隙間 {gap_hi - gap_lo:.4f} で完全に分かれている")
    print("  「x > 0 なら必ず 1」を再現する係数は、傾きが大きいほど尤度が高い。上限がない")

    print("\n  反復を止める場所を変えて当ててみる（IRLS）")
    print("  最大反復   切片        傾き           傾きの SE     逸脱度")
    paths = {}
    for it in (2, 5, 10, 15, 20, 25):
        res = glm.irls(X, y, add_const=False, max_iter=it, tol=1e-14)
        paths[it] = res.b
        print(f"    {it:>3}    {res.b[0]:+8.3f}  {res.b[1]:12.3f}  {res.se[1]:14.1f}"
              f"   {res.deviance:.3e}")
    print("  ← 止めた場所が答えになっている。これは推定ではなく、打ち切りの記録である")
    print("  逸脱度が 0 に向かうのは「完全に当たっている」から。当てはまりの良さは分離の症状")

    # statsmodels に投げるとどうなるか。
    with warnings.catch_warnings(record=True) as caught, np.errstate(over="ignore"):
        warnings.simplefilter("always")
        logit = sm.Logit(y, X).fit(disp=0, maxiter=100)
    kinds = sorted({w.category.__name__ for w in caught})
    z = logit.params[1] / logit.bse[1]
    print(f"\n  statsmodels.Logit  傾き {logit.params[1]:.2f}（SE {logit.bse[1]:.3e}）")
    print(f"  収束したか {logit.mle_retvals['converged']}、反復 {logit.mle_retvals['iterations']} 回、"
          f"出た警告 {', '.join(kinds)}")
    print(f"  傾きの z = {z:.4f}、Wald の p = {logit.pvalues[1]:.4f}")
    print("  ← 係数は事実上の無限大なのに、p 値は「有意でない」と言う。"
          "SE がそれ以上に大きいから。この出力を額面どおりに読んではいけない")

    b_firth = firth_logit(X, y)
    print(f"\n  Firth 罰則版      切片 {b_firth[0]:+.4f} / 傾き {b_firth[1]:.4f}"
          "   ← 有限の答えが返る（大きいが、発散はしない）")
    return x, y, paths, b_firth


def rare_event_trial(rng):
    """まれな事象（p≈0.01）で1回ぶん。最尤推定と Firth の係数を返す。"""
    b_true = np.asarray(B_RARE, dtype=float)
    x = rng.normal(0.0, 1.0, size=N_RARE)
    X = np.column_stack([np.ones(N_RARE), x])
    y = (rng.random(N_RARE) < glm.expit(X @ b_true)).astype(float)
    mle = glm.irls(X, y, add_const=False, max_iter=60)
    firth = firth_logit(X, y)
    return y.sum(), mle.b[0], mle.b[1], firth[0], firth[1]


def rare_event_demo():
    """まれな事象での最尤推定のバイアスを、回して数える。"""
    b0_true, b1_true = B_RARE
    p_center = glm.expit(b0_true)
    print(f"\n--- まれな事象での最尤推定のバイアス（n={N_RARE}, seed={SEED_RARE}, "
          f"真値 [{b0_true}, {b1_true}]）---")
    print(f"  x=0 での事象確率 {p_center:.4f}、平均の事象確率はおよそ "
          f"{np.mean(glm.expit(b0_true + np.random.default_rng(0).normal(size=200_000))):.4f}")

    with sim.Timer(f"  {TRIALS:,} 回"):
        out = sim.repeat(rare_event_trial, trials=TRIALS, seed=SEED_RARE, progress=False)
    events, b0, b1, f0, f1 = out.T

    bad = (np.abs(b0) > DIVERGED) | (np.abs(b1) > DIVERGED)
    print(f"  1試行あたりの事象数 平均 {events.mean():.2f}（最小 {events.min():.0f} / "
          f"最大 {events.max():.0f}）、事象が0件だった試行 {int((events == 0).sum())} 回")
    print(f"  最尤推定が発散した試行（|係数| > {DIVERGED:g}）{int(bad.sum())} 回 "
          f"= {bad.mean():.3%}   ← まれな事象は分離と地続き")

    print("\n  係数の平均（真値からのずれ）")
    print("  推定法               切片            傾き")
    for name, e0, e1 in (("最尤推定", b0, b1), ("Firth 罰則付き", f0, f1)):
        print(f"  {name:<18}{e0.mean():+8.4f} ({e0.mean() - b0_true:+.4f})"
              f"   {e1.mean():+7.4f} ({e1.mean() - b1_true:+.4f})")
    print(f"  最尤推定の切片のバイアス {b0.mean() - b0_true:+.4f}"
          f"（真値の {100 * (b0.mean() - b0_true) / abs(b0_true):+.1f}%）"
          f" → Firth で {f0.mean() - b0_true:+.4f}"
          f"（{100 * (f0.mean() - b0_true) / abs(b0_true):+.1f}%）")
    print("  ← 偏りは主に切片に出る。切片が下に偏るということは、"
          "**事象の確率を系統的に低く見積もる**ということ")
    keep = ~bad
    print(f"  発散した {int(bad.sum())} 回を除くと 切片 {b0[keep].mean() - b0_true:+.4f} / "
          f"傾き {b1[keep].mean() - b1_true:+.4f} まで縮む。"
          "偏りの半分は発散した試行、残り半分は有限のまま下に寄っているぶん")
    print(f"  切片の中央値      最尤 {np.median(b0):+.4f} / Firth {np.median(f0):+.4f}"
          f"（真値 {b0_true:+.1f}）   ← 外れ値に頼らない見方でも下に寄る")
    print(f"  x=0 での予測確率  真値 {p_center:.5f} / 最尤の中央値から "
          f"{glm.expit(np.median(b0)):.5f}（{100 * (glm.expit(np.median(b0)) / p_center - 1):+.1f}%）"
          f" / Firth {glm.expit(np.median(f0)):.5f}")
    return b0, f0, b0_true


def main() -> None:
    plots.setup()
    x, y, paths, b_firth = separation_demo()
    b0, f0, b0_true = rare_event_demo()

    # --- 図 ---
    fig, axes = plots.figure(1, 2, w=1.8, h=1.0)

    ax = axes[0]
    grid = np.linspace(x.min() - 0.2, x.max() + 0.2, 601)
    ax.scatter(x, y, s=8, color=plots.PALETTE["data"], alpha=0.6, lw=0, zorder=3)
    # ラベルは曲線ごとに違う高さ（p = 0.5 / 0.75 / 0.95 の点）に置く。同じ高さに置くと
    # 3本とも x=0 付近に重なって読めない。
    for (it, alpha), p_anchor, dx in (((2, 0.40), 0.50, -34), ((5, 0.65), 0.75, 12),
                                      ((10, 0.95), 0.95, 12)):
        b = paths[it]
        ax.plot(grid, glm.expit(b[0] + b[1] * grid), color=plots.PALETTE["reject"],
                lw=1.1, alpha=alpha, zorder=4)
        x_anchor = (np.log(p_anchor / (1 - p_anchor)) - b[0]) / b[1]
        ax.annotate(f"{it} 反復", xy=(x_anchor, p_anchor), xytext=(dx, 0),
                    textcoords="offset points", fontsize=5.8, va="center",
                    color=plots.PALETTE["reject"], alpha=max(alpha, 0.7))
    ax.plot(grid, glm.expit(b_firth[0] + b_firth[1] * grid),
            color=plots.PALETTE["estimate"], lw=1.3, ls="--", dashes=(4, 2.0), zorder=5)
    ax.annotate("Firth 罰則版", xy=(grid[-1], 1.0), xytext=(-2, -10),
                textcoords="offset points", fontsize=6.0, ha="right",
                color=plots.PALETTE["estimate"])
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$（0/1）と予測確率")
    ax.set_title("反復を重ねるほど急になる（止まらない）")

    ax = axes[1]
    lo, hi = np.quantile(np.r_[b0, f0], [0.005, 0.995])
    bins = np.linspace(lo, hi, 50)
    ax.hist(np.clip(b0, lo, hi), bins=bins, density=True, color=plots.PALETTE["data"],
            alpha=0.55, lw=0)
    ax.hist(np.clip(f0, lo, hi), bins=bins, density=True, histtype="step",
            color=plots.PALETTE["estimate"], lw=1.2, zorder=4)
    ax.axvline(b0.mean(), color=plots.PALETTE["reject"], lw=1.1, ls="--",
               dashes=(4, 2.2), zorder=5)
    ax.annotate(f"最尤の平均 {b0.mean():.2f}", xy=(b0.mean(), 0.98),
                xycoords=("data", "axes fraction"), xytext=(-3, 0),
                textcoords="offset points", fontsize=6.0, ha="right", va="top",
                color=plots.PALETTE["reject"])
    ax.annotate("Firth（枠線）", xy=(b0_true, 0.60), xycoords=("data", "axes fraction"),
                xytext=(4, 0), textcoords="offset points", fontsize=6.0,
                color=plots.PALETTE["estimate"])
    ax.annotate("← 発散した試行は左端に寄せてある", xy=(lo, 0.30),
                xycoords=("data", "axes fraction"), xytext=(3, 0),
                textcoords="offset points", fontsize=5.8, color=plots.PALETTE["ink2"])
    plots.mark_truth(ax, b0_true, f"真値 = {b0_true:g}")
    ax.set_xlabel("切片の推定値 $\\hat{b}_0$")
    ax.set_ylabel("密度")
    ax.set_title(f"まれな事象では切片が下に偏る（{TRIALS:,} 回）")
    fig.tight_layout()
    plots.save(fig, "fig-13-8-separation-and-rare-events.png")


if __name__ == "__main__":
    main()
