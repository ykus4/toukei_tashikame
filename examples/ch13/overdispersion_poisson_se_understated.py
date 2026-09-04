"""過分散 — ポアソンの標準誤差は「平均=分散」を信じているぶんだけ小さい。

ポアソン分布は平均と分散が等しい。この等式はモデルの都合であって、データの性質では
ない。実際の回数データは、たいてい分散のほうが大きい（**過分散**）。そして GLM の
標準誤差は分散関数 $V(\\mu)=\\mu$ から作られるので、分散を過小に見積もったモデルは
標準誤差も過小に出す。**係数そのものはほぼ正しく、区間だけが狭い**——これが過分散の
いちばん厄介なところで、点推定を見ているかぎり異常に気づけない。

分散が平均の3倍になるデータ（ポアソン・ガンマ混合＝負の二項）を作り、ポアソン回帰と
負の二項回帰を何度も当てて、(1) 平均の標準誤差、(2) 推定値の実際のばらつき、
(3) 95%信頼区間の被覆確率を数え上げる。標準誤差が信用できるとは「(1) が (2) に等しい」
という意味である、ということを数字で確かめる。

    uv run python examples/ch13/overdispersion_poisson_se_understated.py
"""

import numpy as np
from scipy import stats

from toukei_tashikame import glm, plots, sim

N, B_TRUE, OVERDISP = 800, (0.5, 0.5), 3.0
SEED, TRIALS = 137, 4_000
Z = stats.norm.ppf(0.975)


def make_data(rng):
    """分散が平均の OVERDISP 倍になる回数データ。ポアソン・ガンマ混合で作る。

    ガンマで平均そのものをばらつかせてからポアソンを引く。混合の結果、
    $\\mathrm{Var}[y] = \\mathrm{OVERDISP}\\times\\mu$ になる。
    """
    b = np.asarray(B_TRUE, dtype=float)
    X = np.column_stack([np.ones(N), rng.normal(0.0, 1.0, size=N)])
    mu = np.exp(X @ b)
    shape = mu / (OVERDISP - 1.0)
    y = rng.poisson(rng.gamma(shape, scale=OVERDISP - 1.0)).astype(float)
    return X, y


def one_trial(rng):
    """1回ぶん。同じデータにポアソンと負の二項を当て、傾きの推定と SE を返す。"""
    X, y = make_data(rng)
    pois = glm.irls(X, y, family="poisson", add_const=False)
    nb = glm.negbin(X, y, add_const=False)
    return pois.b[1], pois.se[1], nb.b[1], nb.se[1], glm.dispersion(pois)


def main() -> None:
    plots.setup()
    b1_true = B_TRUE[1]

    # --- まず1組のデータを覗く ---
    X, y = make_data(np.random.default_rng(SEED))
    pois0 = glm.irls(X, y, family="poisson", add_const=False)
    print(f"--- 13-7 過分散（n={N}, seed={SEED}, 真の傾き {b1_true}, 分散=平均の{OVERDISP:g}倍）---")
    print(f"  y の平均 {y.mean():.4f} / 分散 {y.var(ddof=1):.4f}"
          f"   ← 比 {y.var(ddof=1) / y.mean():.2f}（ポアソンなら 1.00 のはず）")
    print(f"  Pearson χ²/df = {glm.dispersion(pois0):.3f}   ← 当てはめ後の残差で測っても同じ結論")
    print("  この1つの数字だけで、以下の区間がすべて狭すぎることが分かる")

    # --- 回して数える ---
    with sim.Timer(f"  {TRIALS:,} 回"):
        out = sim.repeat(one_trial, trials=TRIALS, seed=SEED, progress=False)
    b_p, se_p, b_n, se_n, disp = out.T

    print(f"\n--- {TRIALS:,} 回まわして数える ---")
    print("  モデル        係数の平均   モデルが言う SE   実際のばらつき(SD)   SE / SD")
    for name, b, se in (("ポアソン", b_p, se_p), ("負の二項", b_n, se_n)):
        sd = b.std(ddof=1)
        print(f"  {name:<12}{b.mean():+.4f}      {se.mean():.4f}           {sd:.4f}"
              f"          {se.mean() / sd:.3f}")
    print(f"  ← 係数はどちらも真値 {b1_true} の近く。**壊れているのは SE だけ**")
    print(f"  ポアソンの SE は負の二項の {se_n.mean() / se_p.mean():.2f} 分の1 "
          f"（{se_n.mean():.4f} / {se_p.mean():.4f}）")

    # 被覆確率。真値を包んだ区間の割合を数える。
    print("\n  95%信頼区間が真値を包んだ割合")
    rates = {}
    for name, b, se in (("ポアソン", b_p, se_p), ("負の二項", b_n, se_n)):
        lo, hi = b - Z * se, b + Z * se
        rate = float(((lo <= b1_true) & (b1_true <= hi)).mean())
        rates[name] = rate
        # 数え上げそのものの誤差。この幅より小さい差は、回し直すだけで動く。
        se_rate = np.sqrt(rate * (1 - rate) / TRIALS)
        print(f"  {name:<12}{rate:.4f} ± {1.96 * se_rate:.4f}   "
              f"（平均の区間幅 {2 * Z * se.mean():.4f}）")
    print(f"  ← 名目 0.95 に対してポアソンは {0.95 - rates['ポアソン']:.4f} 足りない。"
          "「20回に1回外す」つもりが実際は "
          f"{1 / max(1 - rates['ポアソン'], 1e-9):.1f} 回に1回")
    print(f"  Pearson χ²/df の平均 {disp.mean():.3f}（真の過分散 {OVERDISP:g}）。"
          f"√{disp.mean():.2f} = {np.sqrt(disp.mean()):.2f} 倍が、SE の過小の見当になる")
    print(f"  負の二項も {rates['負の二項']:.4f} で名目にわずかに届かない。"
          "過分散パラメータ α を積率法で推定した不確実性が SE に入っていないため")

    # --- 図 ---
    fig, axes = plots.figure(1, 2, w=1.8, h=1.0)

    ax = axes[0]
    ax.hist(b_p, bins=50, density=True, color=plots.PALETTE["data"], alpha=0.55, lw=0)
    grid = np.linspace(b_p.min(), b_p.max(), 400)
    ax.plot(grid, stats.norm.pdf(grid, b1_true, se_p.mean()),
            color=plots.PALETTE["reject"], lw=1.3, zorder=5)
    ax.plot(grid, stats.norm.pdf(grid, b1_true, b_p.std(ddof=1)),
            color=plots.PALETTE["estimate"], lw=1.3, ls="--", dashes=(4, 2.0), zorder=5)
    ax.annotate("ポアソンが主張する分布\n（SE から作った正規）",
                xy=(b1_true, stats.norm.pdf(b1_true, b1_true, se_p.mean())),
                xytext=(6, -4), textcoords="offset points", fontsize=6.0,
                color=plots.PALETTE["reject"], va="top")
    ax.annotate("実際のばらつき", xy=(grid[-1], 0.0), xytext=(-2, 14),
                textcoords="offset points", fontsize=6.0, ha="right",
                color=plots.PALETTE["estimate"])
    plots.mark_truth(ax, b1_true, f"真値 = {b1_true:g}")
    ax.set_xlabel("傾きの推定値 $\\hat{b}_1$")
    ax.set_ylabel("密度")
    ax.set_title("推定値は正しい。細すぎるのは主張のほう")

    ax = axes[1]
    lo, hi = b_p - Z * se_p, b_p + Z * se_p
    missed = plots.coverage_stripes(ax, np.column_stack([lo, hi]), b1_true, n_show=100)
    ax.set_xlabel("ポアソン回帰の 95% 信頼区間")
    ax.set_title(f"最初の100本で {missed} 本が外れ（名目は 5 本）")
    fig.tight_layout()
    plots.save(fig, "fig-13-7-overdispersion-coverage.png")


if __name__ == "__main__":
    main()
