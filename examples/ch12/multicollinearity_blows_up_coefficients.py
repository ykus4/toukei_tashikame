"""多重共線性 — 説明変数どうしが似ていると、係数だけが暴れる。当てはまりは無傷のまま。

真のモデルは $y = x_1 + x_2 + \\varepsilon$ で固定し、$x_1, x_2$ の相関を 0 / 0.5 / 0.9 / 0.99 と
変えて 10,000 回ずつ推定する。相関 0.99 では $\\hat b_1$ の標準偏差が7倍に膨らみ、真値が
+1.0 なのに**符号が逆に出る**回が 8% ある。

それでも残差の大きさ $\\hat\\sigma$ は真値のままで、予測の精度は落ちない。壊れているのは
「どちらの変数の効果か」を
切り分ける能力だけで、和 $b_1 + b_2$ は最後まで正確に推定できている。VIF が言っている
のはこのことで、「予測はできるが説明はできない」という状態の名前である。

    uv run python examples/ch12/multicollinearity_blows_up_coefficients.py
"""

import numpy as np

from toukei_tashikame import plots, regression

N, TRIALS, SEED = 100, 10_000, 127
B1, B2, SIGMA = 1.0, 1.0, 1.0
RHOS = (0.0, 0.5, 0.9, 0.99)
CHUNK = 1_000


def batch(rho: float, trials: int, rng):
    """重回帰を trials 回ぶんまとめて解き、(b1, b2, σ̂, R², 標本相関) を返す。"""
    x1 = rng.normal(0.0, 1.0, size=(trials, N))
    x2 = rho * x1 + np.sqrt(1.0 - rho**2) * rng.normal(0.0, 1.0, size=(trials, N))
    y = B1 * x1 + B2 * x2 + rng.normal(0.0, SIGMA, size=(trials, N))

    x1 -= x1.mean(axis=1, keepdims=True)
    x2 -= x2.mean(axis=1, keepdims=True)
    y -= y.mean(axis=1, keepdims=True)
    s11 = (x1 * x1).sum(axis=1)
    s22 = (x2 * x2).sum(axis=1)
    s12 = (x1 * x2).sum(axis=1)
    s1y = (x1 * y).sum(axis=1)
    s2y = (x2 * y).sum(axis=1)
    syy = (y * y).sum(axis=1)

    det = s11 * s22 - s12**2
    b1 = (s22 * s1y - s12 * s2y) / det
    b2 = (s11 * s2y - s12 * s1y) / det
    rss = syy - b1 * s1y - b2 * s2y
    sigma2 = rss / (N - 3)
    r2 = 1 - rss / syy
    r12 = s12 / np.sqrt(s11 * s22)      # 説明変数どうしの標本相関
    return b1, b2, np.sqrt(sigma2), r2, r12


def main() -> None:
    plots.setup()
    rng = np.random.default_rng(SEED)

    print(f"--- 真の係数 b1={B1}, b2={B2}、n={N}、各条件 {TRIALS:,} 回 ---")
    print("  x1,x2の相関   VIF(理論)  VIF(実測)   b1の平均   b1の標準偏差   無相関比"
          "   符号が逆   b1+b2 の標準偏差    σ̂     R²")
    results = {}
    base_sd = None
    for rho in RHOS:
        parts = [batch(rho, min(CHUNK, TRIALS - i), rng) for i in range(0, TRIALS, CHUNK)]
        b1, b2, sigma_hat, r2, r12 = (np.concatenate([p[j] for p in parts])
                                      for j in range(5))
        sd = b1.std(ddof=1)
        base_sd = base_sd if base_sd is not None else sd
        vif_emp = float(np.mean(1.0 / (1.0 - r12**2)))
        results[rho] = (b1, b2, sd, vif_emp)
        print(f"     {rho:.2f}       {1 / (1 - rho**2):8.2f}   {vif_emp:8.2f}"
              f"   {b1.mean():8.4f}     {sd:8.4f}     {sd / base_sd:6.2f} 倍"
              f"    {float((b1 < 0).mean()):.4f}        {(b1 + b2).std(ddof=1):.4f}"
              f"        {sigma_hat.mean():.4f}  {r2.mean():.4f}")

    print("\n  係数の平均はどの相関でも真値 1.0 に当たる。共線性はバイアスを生まない。")
    print("  壊れるのは精度で、相関 0.99 では標準偏差が"
          f" {results[0.99][2] / results[0.0][2]:.2f} 倍、"
          f"符号が逆になる回が {float((results[0.99][0] < 0).mean()):.1%} ある。")
    print("  一方 b1+b2 の標準偏差はほとんど変わらない。データが答えられるのは")
    print("  「2つ合わせた効果」までで、内訳を聞かれても答える材料がない、ということ。")
    print("  残差の大きさ σ̂ もどの条件でも真の σ=1.0 のまま。予測の精度は無傷である。")
    print("  （R² が上がっていくのは y の全変動が ρ とともに増えるからで、当てはまりの")
    print("   改善ではない。共線性を R² で診断できないことがここに出ている。）")

    # --- 道具（regression.vif）と突き合わせる ---
    x1 = rng.normal(0.0, 1.0, size=N)
    x2 = 0.99 * x1 + np.sqrt(1 - 0.99**2) * rng.normal(0.0, 1.0, size=N)
    v = regression.vif(np.column_stack([x1, x2]), names=["x1", "x2"])
    print(f"\n--- 照合（1標本、相関0.99）regression.vif ---\n{v.round(3).to_string()}")
    print(f"  理論値は 1/(1-0.99²) = {1 / (1 - 0.99**2):.2f}。1標本の VIF 自体もこれだけ揺れる")

    # --- 図 ---
    fig, axes = plots.figure(1, 2, w=2.0, h=1.0)
    pal = plots.PALETTE

    ax = axes[0]
    bins = np.linspace(-2.5, 4.5, 120)
    for rho, alpha in zip(RHOS, (0.25, 0.35, 0.45, 0.60), strict=True):
        ax.hist(results[rho][0], bins=bins, color=pal["estimate"], alpha=alpha, lw=0)
        ax.annotate(f"相関 {rho:.2f}（sd {results[rho][2]:.3f}）",
                    xy=(0.02, 0.95 - 0.08 * RHOS.index(rho)), xycoords="axes fraction",
                    fontsize=6.0, color=pal["estimate"], alpha=min(1.0, alpha + 0.4))
    plots.mark_truth(ax, B1, f"真値 {B1}")
    ax.axvline(0.0, color=pal["reject"], lw=0.9, ls="--", dashes=(4, 2.2))
    ax.annotate("ここより左は符号が逆", xy=(0.0, 0.35), xycoords=("data", "axes fraction"),
                xytext=(-3, 0), textcoords="offset points", ha="right",
                fontsize=6.0, color=pal["reject"])
    ax.set_xlabel("$\\hat{b}_1$")
    ax.set_ylabel(f"{TRIALS:,} 回のうちの回数")
    ax.set_title("① 相関が上がるほど係数が暴れる")

    ax = axes[1]
    vifs = np.array([results[r][3] for r in RHOS])
    sds = np.array([results[r][2] for r in RHOS])
    ax.scatter(vifs, sds, s=18, color=pal["estimate"], zorder=4)
    grid = np.linspace(1.0, vifs.max() * 1.05, 200)
    ax.plot(grid, sds[0] * np.sqrt(grid), color=pal["truth"], lw=1.1, ls="--",
            dashes=(4, 2.0), zorder=3)
    ax.annotate("理論: 標準偏差 ∝ √VIF", xy=(grid[140], sds[0] * np.sqrt(grid[140])),
                xytext=(4, -2), textcoords="offset points", fontsize=6.0, color=pal["truth"])
    for rho, v_, s_ in zip(RHOS, vifs, sds, strict=True):
        ax.annotate(f"ρ={rho:.2f}", xy=(v_, s_), xytext=(3, 4),
                    textcoords="offset points", fontsize=6.0, color=pal["ink2"])
    ax.set_xlabel("VIF（実測）")
    ax.set_ylabel("$\\hat{b}_1$ の標準偏差")
    ax.set_title("② VIF は係数の標準偏差の2乗を測っている")

    fig.tight_layout()
    plots.save(fig, "fig-12-7-vif-coefficient-variance.png")


if __name__ == "__main__":
    main()
