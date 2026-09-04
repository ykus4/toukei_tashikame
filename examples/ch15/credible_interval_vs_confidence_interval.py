"""信用区間と信頼区間は、数字はほぼ同じで、意味している文が違う。

95% 信用区間は「この区間に $p$ が入る確率が 95%」と読んでよい。主語はパラメータで、
区間は動かない。95% 信頼区間はそう読めない。読めるのは「この手続きを繰り返すと、
作った区間の 95% が真値を包む」で、主語は**手続き**であり、動くのは区間のほうである。

同じ標本（n=40）から両方を作ると、端が 0.005 ほどしか違わない。それでも文が違うので、
被覆確率を数えて両方が 95% 付近に着地することまで見ておく。数字が近いことと、
言っていることが同じであることは、別である。

    uv run python examples/ch15/credible_interval_vs_confidence_interval.py
"""

import numpy as np

from toukei_tashikame import bayes, estimate, plots, sim

N, P_TRUE, SEED = 40, 0.28, 157
PRIOR_A, PRIOR_B = 1.0, 1.0     # 一様事前
TRIALS = 10_000
CONF = 0.95


def credible(k: int, a: float = PRIOR_A, b: float = PRIOR_B) -> tuple[float, float]:
    """ベイズの信用区間。Beta(a,b) 事前 → 事後 Beta(a+k, b+n-k) の等裾 95%。"""
    return bayes.beta_binomial(k, N, a, b).interval(CONF)


def one_credible(rng) -> tuple[float, float]:
    return credible(int(rng.binomial(N, P_TRUE)))


def one_credible_strong(rng) -> tuple[float, float]:
    """平均 0.20 の強い事前（100 件ぶん）。真値 0.28 から外れている。"""
    return credible(int(rng.binomial(N, P_TRUE)), 20.0, 80.0)


def one_wilson(rng) -> tuple[float, float]:
    return estimate.ci_prop_wilson(int(rng.binomial(N, P_TRUE)), N, CONF)


def draw(bayes_res, wilson_res, k, cred, wils) -> None:
    fig, axes = plots.figure(1, 3, w=2.2, h=1.0)
    pal = plots.PALETTE

    ax = axes[0]
    grid = np.linspace(0.0, 0.7, 600)
    post = bayes.beta_binomial(k, N, PRIOR_A, PRIOR_B)
    ax.plot(grid, post.pdf(grid), color=pal["posterior"], lw=1.3, zorder=4)
    inside = (grid >= cred[0]) & (grid <= cred[1])
    ax.fill_between(grid[inside], post.pdf(grid[inside]), color=pal["posterior"],
                    alpha=0.22, lw=0, zorder=1)
    plots.mark_truth(ax, P_TRUE, f"真値 = {P_TRUE}")
    ax.annotate(f"信用区間\n[{cred[0]:.3f}, {cred[1]:.3f}]\n（塗った面積が 95%）",
                xy=(0.02, 0.72), xycoords="axes fraction", fontsize=6.0,
                color=pal["estimate"], va="top")
    ax.annotate(f"信頼区間\n[{wils[0]:.3f}, {wils[1]:.3f}]",
                xy=(0.62, 0.40), xycoords="axes fraction", fontsize=6.0,
                color=pal["ink2"], va="top")
    ax.set_xlabel("$p$")
    ax.set_ylabel("事後密度")
    ax.set_title("① 同じ標本から作った2つ")

    for ax, res, title in ((axes[1], bayes_res, "② 信用区間を 100 本"),
                           (axes[2], wilson_res, "③ Wilson 信頼区間を 100 本")):
        missed = plots.coverage_stripes(ax, res.intervals, P_TRUE, n_show=100)
        ax.set_xlim(0.05, 0.55)
        ax.set_xlabel("$p$")
        ax.set_title(f"{title}（外した {missed} 本が赤）")

    fig.suptitle("動いているのは区間のほう。真値は一度も動かない", fontsize=7)
    fig.tight_layout()
    plots.save(fig, "fig-15-7-credible-vs-confidence.png")


def main() -> None:
    plots.setup()
    rng = np.random.default_rng(SEED)
    x = (rng.random(N) < P_TRUE).astype(int)
    k = int(x.sum())

    cred = credible(k)
    wils = estimate.ci_prop_wilson(k, N, CONF)
    post = bayes.beta_binomial(k, N, PRIOR_A, PRIOR_B)

    print(f"--- 1つの標本（n={N}、観測 {k}/{N} = {k / N:.4f}、真値 {P_TRUE}、"
          f"seed={SEED}）---\n")
    print(f"  95% 信用区間（Beta(1,1) 事前）  [{cred[0]:.4f}, {cred[1]:.4f}]"
          f"  幅 {cred[1] - cred[0]:.4f}")
    print(f"  95% Wilson 信頼区間             [{wils[0]:.4f}, {wils[1]:.4f}]"
          f"  幅 {wils[1] - wils[0]:.4f}")
    print(f"  端の差                          下 {abs(cred[0] - wils[0]):.4f} / "
          f"上 {abs(cred[1] - wils[1]):.4f}\n")

    print("  数字は近い。だが言える文は違う:")
    print(f"    信用区間  「$p$ が [{cred[0]:.3f}, {cred[1]:.3f}] に入る確率は 95%」")
    print("              — 事後分布の面積そのもの。この文はこの標本1つで完結する")
    print(f"    信頼区間  「[{wils[0]:.3f}, {wils[1]:.3f}] に $p$ が入る確率は 95%」とは言えない")
    print("              — $p$ は定数なので、入っているか入っていないかのどちらか。")
    print("                95% は、この手続きを繰り返したときの当たり率である\n")

    # 3つとも同じ seed。同じ 10,000 本の標本に別々の手続きを当てる対応のある比較で、
    # 出た差は手続きの差そのものになる（標本の引き直しでは動かない）。
    bayes_res = sim.coverage(one_credible, truth=P_TRUE, trials=TRIALS, seed=SEED,
                             progress=False)
    wilson_res = sim.coverage(one_wilson, truth=P_TRUE, trials=TRIALS, seed=SEED,
                              progress=False)
    strong_res = sim.coverage(one_credible_strong, truth=P_TRUE, trials=TRIALS, seed=SEED,
                              progress=False)

    print(f"--- その「繰り返したときの当たり率」を {TRIALS:,} 回で数える ---\n")
    print(f"{'':<28}{'被覆率':>12}{'±1.96SE':>11}{'平均の幅':>12}")
    for label, res in (("信用区間 Beta(1,1) 事前", bayes_res),
                       ("Wilson 信頼区間", wilson_res),
                       ("信用区間 Beta(20,80) 事前", strong_res)):
        width = float((res.intervals[:, 1] - res.intervals[:, 0]).mean())
        print(f"  {label:<26}{res.rate:>12.4f}{1.96 * res.se:>11.4f}{width:>12.4f}")

    disagree = int((bayes_res.covered != wilson_res.covered).sum())
    print(f"\n  上の2つは同じ {TRIALS:,} 本の標本に当てている。当たり外れが食い違ったのは"
          f" {disagree} 本で、")
    print(f"  被覆率の差は {abs(bayes_res.rate - wilson_res.rate):.4f}。"
          f"シミュレーション誤差 ±{1.96 * bayes_res.se:.4f} より小さい。")
    print("  数字としては、この2つは実質同じものが出ている。\n")

    print("  一様事前の信用区間が頻度論的な被覆も満たすのは偶然ではない。Beta(1,1) は")
    print("  情報をほとんど足しておらず、事後は尤度そのものに近い形をしているからである。")
    print(f"  一方、真値 0.28 から外れた強い事前 Beta(20,80) を入れると被覆は "
          f"{strong_res.rate:.4f} に落ちる。")
    print("  これは欠陥ではなく「事前の主張を込めた」ことの当然の帰結で、そのぶん区間は")
    print(f"  狭い（幅 {float((strong_res.intervals[:, 1] - strong_res.intervals[:, 0]).mean()):.4f}）。"
          "答えている問いが違う、とはこういうことである。\n")

    print(f"  参考: 事後 {post}")
    draw(bayes_res, wilson_res, k, cred, wils)


if __name__ == "__main__":
    main()
