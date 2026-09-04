"""正規方程式を5行で書き、lstsq と statsmodels に同じ数字が出ることを確かめる。

回帰の出力にある「係数・標準誤差・$R^2$」は、どれも $(X^\\top X)\\hat b = X^\\top y$ と
その分散 $\\sigma^2 (X^\\top X)^{-1}$ の2つから出てくる。ライブラリの `summary()` は
この2つを表に組み直しているだけで、中で別のことをしているわけではない。

一度自分で書いてから3つを突き合わせると、差が $10^{-15}$ の桁（倍精度の丸め）でしか
出ないことが分かる。ここで一致を確認しておけば、以降の章でライブラリの出力を読むとき、
どの数字がどの式から来たのかを毎回たどれる。

    uv run python examples/ch12/ols_normal_equations_vs_lstsq.py
"""

import numpy as np
import statsmodels.api as sm

from toukei_tashikame import datasets, regression

N, SEED = 200, 122
B_TRUE = (1.0, 2.0, -1.0, 0.5)     # 先頭が切片。説明変数は3本


def by_normal_equations(X: np.ndarray, y: np.ndarray):
    """正規方程式をそのまま解く。これが回帰の全部である。"""
    b = np.linalg.solve(X.T @ X, X.T @ y)          # ① 係数
    resid = y - X @ b                              # ② 残差
    sigma2 = resid @ resid / (X.shape[0] - X.shape[1])   # ③ 誤差分散
    se = np.sqrt(np.diag(sigma2 * np.linalg.inv(X.T @ X)))   # ④ 標準誤差
    r2 = 1 - resid @ resid / ((y - y.mean()) ** 2).sum()     # ⑤ 決定係数
    return b, se, float(r2)


def by_lstsq(X: np.ndarray, y: np.ndarray):
    """np.linalg.lstsq。中身は QR 分解で、逆行列を作らない。"""
    b, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ b
    sigma2 = resid @ resid / (X.shape[0] - X.shape[1])
    se = np.sqrt(np.diag(sigma2 * np.linalg.inv(X.T @ X)))
    r2 = 1 - resid @ resid / ((y - y.mean()) ** 2).sum()
    return b, se, float(r2)


def main() -> None:
    X, y, b_true = datasets.regression_data(N, b=B_TRUE, sigma=1.0, seed=SEED)

    b_ne, se_ne, r2_ne = by_normal_equations(X, y)
    b_ls, se_ls, r2_ls = by_lstsq(X, y)
    sm_res = sm.OLS(y, X).fit()
    lib = regression.ols(X, y)     # 本書の道具（中身は①〜⑤と同じ）

    print(f"--- n={N}、説明変数3本、真の係数 {np.round(b_true, 3).tolist()} ---")
    print(" " * 18 + "     切片       x1       x2       x3")
    for label, b in [("正規方程式      ", b_ne), ("lstsq           ", b_ls),
                     ("statsmodels     ", sm_res.params),
                     ("regression.ols  ", lib.b)]:
        print("  " + label + "".join(f"{v:9.4f}" for v in b))
    print("  " + "-" * 54)
    print("  " + "真値            " + "".join(f"{v:9.4f}" for v in b_true))

    bs = np.array([b_ne, b_ls, sm_res.params, lib.b])
    ses = np.array([se_ne, se_ls, sm_res.bse, lib.se])
    r2s = np.array([r2_ne, r2_ls, sm_res.rsquared, lib.r2])
    print(f"\n  4実装の係数の最大絶対差     {np.abs(bs - bs[0]).max():.3e}")
    print(f"  4実装の標準誤差の最大絶対差 {np.abs(ses - ses[0]).max():.3e}")
    print(f"  4実装の R² の最大絶対差     {np.abs(r2s - r2s[0]).max():.3e}")
    print("  ← 差は倍精度の丸めの桁。4つは同じ式を別の順で計算しているだけである")

    print("\n--- 標準誤差と R²（正規方程式）---")
    print("            標準誤差      t値        p値")
    for name, se, t, p in zip(lib.names, se_ne, lib.t, lib.pvalues, strict=True):
        print(f"  {name:<8}{se:10.4f}{t:9.3f}{p:11.3g}")
    print(f"  R² = {r2_ne:.4f}   調整済み R² = {lib.r2_adj:.4f}   σ̂ = {np.sqrt(lib.sigma2):.4f}")

    print("\n  σ̂ が真の σ=1.0 の付近に来ていること、係数が真値の付近に来ていることを見る。")
    print("  ずれの大きさは標準誤差の1〜2倍に収まっている——それが標準誤差の意味である。")


if __name__ == "__main__":
    main()
