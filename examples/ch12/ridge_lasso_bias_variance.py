"""罰則つき回帰の得失を、予測MSEをバイアス²とバリアンスに割って測る。

n=50 に対して説明変数20本、そのうち真に効いているのは3本だけ、という設定。OLS はどの
係数も偏りなく推定するが、20本ぶんの雑音をそのまま予測に持ち込むのでバリアンスが大きい。
罰則をかけると係数は真値より小さめに縮み（バイアスが出る）、代わりにばらつきが減る。

$\\text{MSE} = \\text{バイアス}^2 + \\text{バリアンス}$ の内訳を数え上げると、罰則が
「わざと偏らせて、そのぶん安定させる」取引だということが数字で見える。不偏であることは
目的ではない、という第6-3節の主張が、予測誤差の分解として現れる。

バイアス²とバリアンスは、固定したテスト点での予測 $\\hat f(x)$ の分布から測る。
$\\text{バイアス}^2 = \\overline{(\\mathbb{E}[\\hat f] - f)^2}$、
$\\text{バリアンス} = \\overline{\\mathrm{Var}[\\hat f]}$ で、どちらも真の $f$ を
知っている合成データでしか数えられない。

    uv run python examples/ch12/ridge_lasso_bias_variance.py
"""

import unicodedata

import numpy as np

from toukei_tashikame import plots, regression, sim

N, P, SEED = 50, 20, 1210
TRIALS = 2_000          # OLS・リッジ・λ=0.15 の Lasso はこの回数
SWEEP_TRIALS = 300      # Lasso の λ 掃引だけは重いので回数を落とす
N_TEST = 200            # バイアス²とバリアンスを測る固定のテスト点
SIGMA = 1.0

B_TRUE = np.zeros(P + 1)
B_TRUE[0] = 1.0
B_TRUE[1:4] = (1.5, -2.0, 1.0)      # 効いているのは先頭3本だけ。残り17本は真に0

RIDGE_LAMS = (0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0)
LASSO_LAMS = (0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.45)
RIDGE_PICK, LASSO_PICK = 1.0, 0.15


def pad(s: str, width: int) -> str:
    """全角を2桁と数えて右に詰める。日本語の見出しがある表の桁を揃えるためだけの補助。"""
    w = sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in s)
    return s + " " * max(width - w, 0)


def test_points(rng) -> tuple[np.ndarray, np.ndarray]:
    """テスト点と、そこでの真の $f(x)$。訓練のたびに動かさない。"""
    x_test = np.column_stack([np.ones(N_TEST), rng.normal(0.0, 1.0, size=(N_TEST, P))])
    return x_test, x_test @ B_TRUE


def train(rng) -> tuple[np.ndarray, np.ndarray]:
    """訓練データを1組。y は先頭3本だけで決まり、残り17本は無関係な列である。"""
    X = np.column_stack([np.ones(N), rng.normal(0.0, 1.0, size=(N, P))])
    y = X @ B_TRUE + rng.normal(0.0, SIGMA, size=N)
    return X, y


def decompose(preds: np.ndarray, f: np.ndarray) -> tuple[float, float, float]:
    """(バイアス², バリアンス, MSE)。preds は (試行数, テスト点数)。"""
    bias2 = float(((preds.mean(axis=0) - f) ** 2).mean())
    var = float(preds.var(axis=0).mean())
    return bias2, var, bias2 + var


def main() -> None:
    plots.setup()
    rng = np.random.default_rng(SEED)
    x_test, f_true = test_points(rng)

    with sim.Timer(f"OLS・リッジ{len(RIDGE_LAMS)}点・Lasso {TRIALS:,} 回"):
        ols_pred = np.empty((TRIALS, N_TEST))
        ridge_pred = {lam: np.empty((TRIALS, N_TEST)) for lam in RIDGE_LAMS}
        lasso_pred = np.empty((TRIALS, N_TEST))
        coefs = {"OLS": [], "リッジ": [], "Lasso": []}
        for i in range(TRIALS):
            X, y = train(rng)
            b_ols = np.linalg.solve(X.T @ X, X.T @ y)
            ols_pred[i] = x_test @ b_ols
            for lam in RIDGE_LAMS:
                ridge_pred[lam][i] = x_test @ regression.ridge(X, y, lam)
            b_lasso = regression.lasso(X, y, LASSO_PICK)
            lasso_pred[i] = x_test @ b_lasso
            coefs["OLS"].append(b_ols)
            coefs["リッジ"].append(regression.ridge(X, y, RIDGE_PICK))
            coefs["Lasso"].append(b_lasso)
    coefs = {k: np.array(v) for k, v in coefs.items()}

    print(f"\n--- n={N}、説明変数 {P} 本（真に効くのは3本、係数 "
          f"{B_TRUE[1:4].tolist()}）、σ={SIGMA}、{TRIALS:,} 回 ---")
    print("  予測MSE は固定した 200 個のテスト点での平均。誤差項の分散 σ²=1.0 は含まない")
    print("\n  手法                     予測MSE   ＝ バイアス²  ＋ バリアンス   OLS比")
    rows = [("OLS（罰則なし）", ols_pred),
            (f"リッジ（λ={RIDGE_PICK}）", ridge_pred[RIDGE_PICK]),
            (f"Lasso（λ={LASSO_PICK}）", lasso_pred)]
    base = decompose(ols_pred, f_true)[2]
    summary = {}
    for name, preds in rows:
        bias2, var, mse = decompose(preds, f_true)
        summary[name] = (bias2, var, mse)
        print(f"  {pad(name, 24)}{mse:9.4f}   {bias2:9.4f}   {var:11.4f}"
              f"   {mse / base:6.2f} 倍")

    print("\n  OLS はバイアスがほぼ 0。不偏であることは数字に出ている。")
    print("  それでも予測は一番悪い。20本ぶんの推定誤差が、そのまま予測のばらつきになる。")
    print("  罰則はバイアスを買ってバリアンスを売る取引で、この設定では割に合っている。")

    print(f"\n--- 係数はどこまで縮むか（{TRIALS:,} 回の平均）---")
    print("  手法        b1(真1.5)  b2(真-2.0)  b3(真1.0)   真に0の17本の平均絶対値"
          "   ちょうど0の割合")
    for name in ("OLS", "リッジ", "Lasso"):
        c = coefs[name]
        zeros_abs = np.abs(c[:, 4:]).mean()
        exact_zero = float((c[:, 1:] == 0.0).mean())
        print(f"  {pad(name, 12)}{c[:, 1].mean():8.4f}{c[:, 2].mean():11.4f}"
              f"{c[:, 3].mean():10.4f}{zeros_abs:19.4f}{exact_zero:20.4f}")
    n_selected = (coefs["Lasso"][:, 1:] != 0.0).sum(axis=1)
    hit3 = (coefs["Lasso"][:, 1:4] != 0.0).sum(axis=1)
    print(f"  Lasso が選んだ変数の数は平均 {n_selected.mean():.2f} 本"
          f"（真は3本、中央値 {np.median(n_selected):.0f}）")
    print(f"  真の3本をすべて残せた割合 {float((hit3 == 3).mean()):.4f}")
    print("  リッジは 0 にしない。縮めるだけなので、変数選択はしていない。")

    # --- λ を振って、バイアス²とバリアンスの取引の形を見る ---
    ridge_curve = np.array([decompose(ridge_pred[lam], f_true) for lam in RIDGE_LAMS])

    with sim.Timer(f"Lasso の λ 掃引（{len(LASSO_LAMS)}点 × {SWEEP_TRIALS} 回）"):
        rng2 = np.random.default_rng(SEED + 1)
        lasso_sweep = {lam: np.empty((SWEEP_TRIALS, N_TEST)) for lam in LASSO_LAMS}
        for i in range(SWEEP_TRIALS):
            X, y = train(rng2)
            for lam in LASSO_LAMS:
                lasso_sweep[lam][i] = x_test @ regression.lasso(X, y, lam)
    lasso_curve = np.array([decompose(lasso_sweep[lam], f_true) for lam in LASSO_LAMS])

    print(f"\n--- λ を振る（リッジは {TRIALS:,} 回、Lasso は {SWEEP_TRIALS} 回）---")
    print("  リッジ  λ:  " + "".join(f"{lam:8.2f}" for lam in RIDGE_LAMS))
    print("          MSE:" + "".join(f"{m:8.4f}" for m in ridge_curve[:, 2]))
    print("  Lasso   λ:  " + "".join(f"{lam:8.2f}" for lam in LASSO_LAMS))
    print("          MSE:" + "".join(f"{m:8.4f}" for m in lasso_curve[:, 2]))
    r_best = RIDGE_LAMS[int(np.argmin(ridge_curve[:, 2]))]
    l_best = LASSO_LAMS[int(np.argmin(lasso_curve[:, 2]))]
    print(f"\n  最小になる λ はリッジで {r_best}（MSE {ridge_curve[:, 2].min():.4f}）、"
          f"Lasso で {l_best}（MSE {lasso_curve[:, 2].min():.4f}）")
    print("  真の係数が3本しかない（疎な）設定なので、0にできる Lasso のほうが強い。")
    print("  真の係数が全部小さく効いている設定では逆になる。どちらが良いかはデータ次第で、")
    print("  「罰則をかければ良くなる」という話ではない。")

    # --- 図 ---
    fig, axes = plots.figure(1, 2, w=2.0, h=1.0, sharey=True)
    pal = plots.PALETTE

    # 注釈の位置だけはパネルごとに手で決める。凡例を置かない分、線の隙間に文字を通す
    panels = [
        {"ax": axes[0], "lams": RIDGE_LAMS, "curve": ridge_curve, "trials": TRIALS,
         "name": "① リッジ — 縮めるが0にはしない", "total_at": 5, "bias_at": -2,
         "ols_x": 0.99, "ols_ha": "right"},
        {"ax": axes[1], "lams": LASSO_LAMS, "curve": lasso_curve, "trials": SWEEP_TRIALS,
         "name": "② Lasso — 0にするので変数も選ぶ", "total_at": 1, "bias_at": -2,
         "ols_x": 0.02, "ols_ha": "left"},
    ]
    for pn in panels:
        ax, curve = pn["ax"], pn["curve"]
        x = np.asarray(pn["lams"], dtype=float)
        ax.plot(x, curve[:, 2], color=pal["estimate"], lw=1.5)
        ax.plot(x, curve[:, 1], color=pal["data"], lw=1.1, ls="--", dashes=(4, 2.0))
        ax.plot(x, curve[:, 0], color=pal["alt"], lw=1.1, ls=":", dashes=(1, 1.6))

        j = int(np.argmin(curve[:, 2]))
        ax.scatter([x[j]], [curve[j, 2]], s=22, color=pal["estimate"], zorder=5)

        ax.axhline(base, color=pal["ink2"], lw=0.8, ls="--", dashes=(5, 2.5))
        ax.annotate(f"OLS の MSE {base:.3f}", xy=(pn["ols_x"], base),
                    xycoords=("axes fraction", "data"), xytext=(0, 3),
                    textcoords="offset points", ha=pn["ols_ha"], va="bottom",
                    fontsize=6.0, color=pal["ink2"])

        k = pn["total_at"]
        ax.annotate("合計 MSE", xy=(x[k], curve[k, 2]), xytext=(2, 8),
                    textcoords="offset points", fontsize=6.0, color=pal["estimate"])
        ax.annotate("バリアンス", xy=(x[-1], curve[-1, 1]), xytext=(-2, -4),
                    textcoords="offset points", ha="right", va="top", fontsize=6.0,
                    color=pal["data"])
        m = pn["bias_at"]
        ax.annotate("バイアス²", xy=(x[m], curve[m, 0]), xytext=(3, -4),
                    textcoords="offset points", va="top", fontsize=6.0, color=pal["alt"])

        ax.set_xlabel(f"罰則の強さ λ（{pn['trials']:,} 回の平均）")
        ax.set_title(f"{pn['name']}（最小は λ={x[j]:g}、MSE {curve[j, 2]:.3f}）")
    axes[0].set_ylabel("固定テスト点での予測MSE")
    axes[0].set_ylim(0.0, 1.6)

    fig.tight_layout()
    plots.save(fig, "fig-12-10-ridge-bias-variance.png")


if __name__ == "__main__":
    main()
