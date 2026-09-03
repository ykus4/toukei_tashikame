"""回帰。正規方程式を自分で解くところから始める。

``statsmodels`` を呼べば済む。それでも一度自分で書くのは、回帰の出力にある
「係数・標準誤差・t・p・R²」が**すべて一つの式から出てくる**ことを見るため。
$(X^\\top X)^{-1} X^\\top y$ と、その分散 $\\sigma^2 (X^\\top X)^{-1}$ の2つしかない。

:func:`stepwise` は**偽の変数を拾う様子を見せるために**置いてある。実務で使うための
関数ではない。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

__all__ = [
    "OlsResult", "cooks_distance", "dummy", "interaction", "lasso", "leverage",
    "ols", "ols_summary", "residual_diagnostics", "ridge", "stepwise", "vif",
]


@dataclass(frozen=True)
class OlsResult:
    """最小二乗の結果。表に出る数字はすべてここから作れる。"""

    b: np.ndarray
    se: np.ndarray
    t: np.ndarray
    pvalues: np.ndarray
    r2: float
    r2_adj: float
    resid: np.ndarray
    fitted: np.ndarray
    cov: np.ndarray
    X: np.ndarray
    y: np.ndarray
    names: list[str]

    @property
    def n(self) -> int:
        return self.X.shape[0]

    @property
    def k(self) -> int:
        """説明変数の数（切片を含む）。"""
        return self.X.shape[1]

    @property
    def df_resid(self) -> int:
        return self.n - self.k

    @property
    def sigma2(self) -> float:
        return float((self.resid**2).sum() / self.df_resid)

    def conf_int(self, conf: float = 0.95) -> np.ndarray:
        half = stats.t.ppf(0.5 + conf / 2, self.df_resid) * self.se
        return np.column_stack([self.b - half, self.b + half])


def ols(X, y, add_const: bool = True, names: list[str] | None = None) -> OlsResult:
    """最小二乗。正規方程式 $(X^\\top X)\\hat\\beta = X^\\top y$ を解くだけ。

    逆行列を作らず ``solve`` に渡す。数値的に安定で、しかも「逆行列を求める問題」では
    なく「連立方程式を解く問題」だという見方に合っている。
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    if X.ndim == 1:
        X = X[:, None]
    if add_const and not np.allclose(X[:, 0], 1.0):
        X = np.column_stack([np.ones(len(X)), X])
    if names is None:
        names = ["const"] + [f"x{i}" for i in range(1, X.shape[1])]

    xtx = X.T @ X
    b = np.linalg.solve(xtx, X.T @ y)
    fitted = X @ b
    resid = y - fitted
    n, k = X.shape
    sigma2 = (resid**2).sum() / (n - k)
    cov = sigma2 * np.linalg.inv(xtx)
    se = np.sqrt(np.diag(cov))
    t = b / se
    p = 2 * stats.t.sf(np.abs(t), df=n - k)

    ss_res = float((resid**2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot
    # 自由度で罰する。説明変数を足せば R² は必ず上がるので、上がった分が
    # 偶然かどうかを見るには調整済みのほうを読む。
    r2_adj = 1 - (1 - r2) * (n - 1) / (n - k)

    return OlsResult(b=b, se=se, t=t, pvalues=p, r2=float(r2), r2_adj=float(r2_adj),
                     resid=resid, fitted=fitted, cov=cov, X=X, y=y, names=names)


def ols_summary(res: OlsResult, conf: float = 0.95) -> str:
    """statsmodels の ``summary()`` に対応する自前の表。"""
    ci = res.conf_int(conf)
    lines = [
        f"OLS  n={res.n}  k={res.k}  df_resid={res.df_resid}",
        f"R² = {res.r2:.4f}   調整済み R² = {res.r2_adj:.4f}   σ̂ = {np.sqrt(res.sigma2):.4f}",
        "",
        f"{'':<10}{'係数':>10}{'標準誤差':>12}{'t':>9}{'p':>10}"
        f"{f'[{conf:.0%} 区間]':>22}",
    ]
    for i, name in enumerate(res.names):
        lines.append(
            f"{name:<10}{res.b[i]:>10.4f}{res.se[i]:>12.4f}{res.t[i]:>9.3f}"
            f"{res.pvalues[i]:>10.4g}   [{ci[i, 0]:>7.3f}, {ci[i, 1]:>7.3f}]"
        )
    return "\n".join(lines)


def leverage(res: OlsResult) -> np.ndarray:
    """てこ比。ハット行列の対角。「その点が自分の予測値をどれだけ引っぱるか」。"""
    X = res.X
    xtx_inv = np.linalg.inv(X.T @ X)
    return np.einsum("ij,jk,ik->i", X, xtx_inv, X)


def cooks_distance(res: OlsResult) -> np.ndarray:
    """Cook の距離。その1点を落としたら当てはめ全体がどれだけ動くか。

    てこ比が高いだけでは影響力とは言えない。残差も大きいときに初めて効く。
    """
    h = leverage(res)
    return (res.resid**2 / (res.k * res.sigma2)) * (h / (1 - h) ** 2)


def residual_diagnostics(res: OlsResult, title: str = ""):
    """残差の4枚組（残差 / Q-Q / スケール-ロケーション / 影響力）。

    1枚では足りない。等分散性は残差プロット、正規性はQ-Q、分散の傾向はスケール-
    ロケーション、外れ値の影響は Cook の距離と、破れ方ごとに見える図が違う。
    """
    from . import plots

    fig, axes = plots.figure(2, 2, h=1.7, w=1.6)
    pal = plots.PALETTE

    ax = axes[0, 0]
    ax.scatter(res.fitted, res.resid, s=8, color=pal["data"], lw=0)
    ax.axhline(0, color=pal["truth"], lw=1.0)
    ax.set_xlabel("当てはめ値")
    ax.set_ylabel("残差")
    ax.set_title("① 残差 vs 当てはめ — 曲がりと分散の変化")

    plots.qq(axes[0, 1], res.resid)
    axes[0, 1].set_title("② Q-Q — 残差の正規性")

    ax = axes[1, 0]
    ax.scatter(res.fitted, np.sqrt(np.abs(res.resid / np.sqrt(res.sigma2))), s=8,
               color=pal["data"], lw=0)
    ax.set_xlabel("当てはめ値")
    ax.set_ylabel("√|標準化残差|")
    ax.set_title("③ スケール-ロケーション — 等分散性")

    ax = axes[1, 1]
    d = cooks_distance(res)
    ax.vlines(np.arange(res.n), 0, d, color=pal["data"], lw=0.8)
    thresh = 4 / res.n
    ax.axhline(thresh, color=pal["truth"], lw=1.0, ls="--", dashes=(4, 2.2))
    ax.annotate(f"目安 4/n = {thresh:.3f}", xy=(0.98, thresh), xycoords=("axes fraction", "data"),
                ha="right", va="bottom", fontsize=6.0, color=pal["truth"])
    ax.set_xlabel("観測番号")
    ax.set_ylabel("Cookの距離")
    ax.set_title("④ 影響力 — 1点で結論が動くか")

    if title:
        fig.suptitle(title, fontsize=8)
    fig.tight_layout()
    return fig


def vif(X, names: list[str] | None = None) -> pd.Series:
    """分散拡大係数。「その列が他の列からどれだけ予測できてしまうか」。

    VIF = 1/(1-R²_j)。10 を超えたら共線性を疑う、という慣習があるが、閾値そのものに
    根拠はない。係数の符号が入れ替わるかどうかを見るほうが実務的である。
    """
    X = np.asarray(X, dtype=float)
    if np.allclose(X[:, 0], 1.0):
        X = X[:, 1:]
        if names:
            names = names[1:]
    names = names or [f"x{i + 1}" for i in range(X.shape[1])]
    out = {}
    for j in range(X.shape[1]):
        others = np.delete(X, j, axis=1)
        r2 = ols(others, X[:, j]).r2
        out[names[j]] = 1.0 / (1.0 - r2) if r2 < 1 else np.inf
    return pd.Series(out, name="VIF")


def dummy(s, drop_first: bool = True) -> pd.DataFrame:
    """カテゴリをダミー変数に。``drop_first`` は基準カテゴリを落とす。

    落とさないと切片と完全に共線になる（ダミー変数の罠）。落とした水準が基準になり、
    他の係数は「基準との差」として読む。
    """
    return pd.get_dummies(pd.Series(s), drop_first=drop_first, dtype=float)


def interaction(a, b) -> np.ndarray:
    """交互作用項。単なる積だが、解釈は「片方の効果がもう片方で変わる」。"""
    return np.asarray(a, dtype=float) * np.asarray(b, dtype=float)


def stepwise(X, y, criterion: str = "p", threshold: float = 0.05,
             names: list[str] | None = None) -> list[int]:
    """前向き変数選択。**偽の変数を拾う様子を見せるために置いてある。**

    y と無関係な列を大量に混ぜて走らせると、いくつも「有意」に選ばれる。選ぶという
    行為自体が多重比較であり、選んだあとの p 値はもう p 値として読めない。
    実務でモデルを選ぶ道具として使ってはいけない。
    """
    X = np.asarray(X, dtype=float)
    if np.allclose(X[:, 0], 1.0):
        X = X[:, 1:]
    n, k = X.shape
    selected: list[int] = []
    remaining = list(range(k))

    while remaining:
        best, best_score = None, np.inf
        for j in remaining:
            cols = [*selected, j]
            res = ols(X[:, cols], y)
            score = res.pvalues[-1] if criterion == "p" else (
                n * np.log((res.resid**2).sum() / n) + 2 * (len(cols) + 1))
            if score < best_score:
                best, best_score = j, score
        if criterion == "p" and best_score >= threshold:
            break
        if criterion == "aic" and selected:
            current = ols(X[:, selected], y)
            aic_now = n * np.log((current.resid**2).sum() / n) + 2 * (len(selected) + 1)
            if best_score >= aic_now:
                break
        selected.append(best)
        remaining.remove(best)
    return selected


def ridge(X, y, lam: float, add_const: bool = True) -> np.ndarray:
    """リッジ回帰。$(X^\\top X + \\lambda I)^{-1} X^\\top y$ の閉じた式。

    切片は罰しない。罰すると、y の中心をどこに取るかで答えが変わってしまう。
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    if add_const and not np.allclose(X[:, 0], 1.0):
        X = np.column_stack([np.ones(len(X)), X])
    pen = lam * np.eye(X.shape[1])
    pen[0, 0] = 0.0
    return np.linalg.solve(X.T @ X + pen, X.T @ y)


def lasso(X, y, lam: float, max_iter: int = 1000, tol: float = 1e-8) -> np.ndarray:
    """Lasso を座標降下で。1変数ずつ、軟しきい値で潰していく。

    リッジと違って閉じた式が無い。係数がちょうど 0 になるのは、罰則が原点で
    折れているから——それが「選択もする」と言われる理由である。
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    if np.allclose(X[:, 0], 1.0):
        X = X[:, 1:]
    n, k = X.shape
    mx, sx = X.mean(0), X.std(0)
    Z = (X - mx) / sx
    yc = y - y.mean()

    b = np.zeros(k)
    for _ in range(max_iter):
        b_old = b.copy()
        for j in range(k):
            r = yc - Z @ b + Z[:, j] * b[j]
            rho = Z[:, j] @ r / n
            # 軟しきい値作用素。|rho| が lam 以下なら 0 に落ちる。
            b[j] = np.sign(rho) * max(abs(rho) - lam, 0.0) / (Z[:, j] @ Z[:, j] / n)
        if np.max(np.abs(b - b_old)) < tol:
            break
    coef = b / sx
    return np.r_[y.mean() - mx @ coef, coef]
