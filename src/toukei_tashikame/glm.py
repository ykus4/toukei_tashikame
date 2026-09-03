"""一般化線形モデルを、1本の IRLS で書く。

GLM は3つの部品でできている——**確率分布・線形予測子・リンク関数**。ロジスティック回帰と
ポアソン回帰は、この3つのうち2つを差し替えただけの同じものである。それをコードで示すのが
このモジュールの目的で、だから :func:`irls` は1本しかない。``family`` を変えるだけで
両方になる。

反復重み付き最小二乗（IRLS）は、毎回の反復で作業応答 $z$ と重み $W$ を作り直して
重み付き最小二乗を解く、を繰り返すだけ。中身は :mod:`regression` の正規方程式と同じ。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

__all__ = [
    "GlmResult", "auc", "confusion", "dispersion", "expit", "irls", "log_link",
    "logit", "negbin", "odds_ratio_table", "roc",
]


def logit(p):
    """ロジット。確率を実数直線へ写す。"""
    p = np.asarray(p, dtype=float)
    return np.log(p / (1.0 - p))


def expit(x):
    """ロジットの逆。実数を (0, 1) へ戻す。数値的に安全な形で書く。"""
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    e = np.exp(x[~pos])
    out[~pos] = e / (1.0 + e)
    return out


def log_link(mu):
    """対数リンク。正の平均を実数直線へ。"""
    return np.log(np.asarray(mu, dtype=float))


@dataclass(frozen=True)
class GlmResult:
    """IRLS の結果。収束の軌跡も持つ。"""

    b: np.ndarray
    se: np.ndarray
    z: np.ndarray
    pvalues: np.ndarray
    deviance: float
    n_iter: int
    family: str
    X: np.ndarray
    y: np.ndarray
    mu: np.ndarray
    names: list[str]
    history: list[np.ndarray] = field(default_factory=list)

    @property
    def df_resid(self) -> int:
        return self.X.shape[0] - self.X.shape[1]

    def conf_int(self, conf: float = 0.95) -> np.ndarray:
        half = stats.norm.ppf(0.5 + conf / 2) * self.se
        return np.column_stack([self.b - half, self.b + half])

    def predict(self, X=None) -> np.ndarray:
        X = self.X if X is None else np.asarray(X, dtype=float)
        eta = X @ self.b
        return expit(eta) if self.family == "binomial" else np.exp(eta)


def _family_parts(family: str):
    """(平均関数, 分散関数, 逸脱度) を返す。GLM の3部品のうち2つがここ。"""
    if family == "binomial":
        def mean(eta):
            return expit(eta)

        def var(mu):
            return mu * (1 - mu)

        def dev(y, mu):
            eps = 1e-10
            return 2 * np.sum(y * np.log((y + eps) / (mu + eps))
                              + (1 - y) * np.log((1 - y + eps) / (1 - mu + eps)))
        return mean, var, dev

    if family == "poisson":
        def mean(eta):
            return np.exp(np.clip(eta, -700, 700))

        def var(mu):
            return mu

        def dev(y, mu):
            eps = 1e-10
            return 2 * np.sum(np.where(y > 0, y * np.log((y + eps) / mu), 0.0) - (y - mu))
        return mean, var, dev

    raise ValueError(f"unknown family: {family}")


def irls(X, y, family: str = "binomial", offset=None, tol: float = 1e-8,
         max_iter: int = 25, add_const: bool = True,
         names: list[str] | None = None) -> GlmResult:
    """反復重み付き最小二乗。ロジスティックもポアソンもこの1本で解く。

    毎回、作業応答 $z = \\eta + (y-\\mu)/\\mu'$ と重み $W = \\mu'^2/V(\\mu)$ を作り、
    重み付き最小二乗を解く。収束したら、そのときの $(X^\\top W X)^{-1}$ が
    そのまま係数の分散共分散になる。
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    if X.ndim == 1:
        X = X[:, None]
    if add_const and not np.allclose(X[:, 0], 1.0):
        X = np.column_stack([np.ones(len(X)), X])
    if names is None:
        names = ["const"] + [f"x{i}" for i in range(1, X.shape[1])]
    off = np.zeros(len(y)) if offset is None else np.asarray(offset, dtype=float)

    mean_fn, var_fn, dev_fn = _family_parts(family)
    b = np.zeros(X.shape[1])
    history = [b.copy()]
    n_iter = 0

    for n_iter in range(1, max_iter + 1):  # noqa: B007 — 抜けた時点の値を残す
        eta = X @ b + off
        mu = mean_fn(eta)
        v = np.clip(var_fn(mu), 1e-10, None)
        # 正準リンクなので dmu/deta = V(mu)。作業応答と重みが簡単になる。
        z = eta - off + (y - mu) / v
        w = v
        xtwx = X.T @ (X * w[:, None])
        b_new = np.linalg.solve(xtwx, X.T @ (w * z))
        history.append(b_new.copy())
        if np.max(np.abs(b_new - b)) < tol:
            b = b_new
            break
        b = b_new

    eta = X @ b + off
    mu = mean_fn(eta)
    v = np.clip(var_fn(mu), 1e-10, None)
    cov = np.linalg.inv(X.T @ (X * v[:, None]))
    se = np.sqrt(np.diag(cov))
    zstat = b / se
    return GlmResult(b=b, se=se, z=zstat, pvalues=2 * stats.norm.sf(np.abs(zstat)),
                     deviance=float(dev_fn(y, mu)), n_iter=n_iter, family=family,
                     X=X, y=y, mu=mu, names=names, history=history)


def odds_ratio_table(res: GlmResult, names: list[str] | None = None,
                     conf: float = 0.95) -> pd.DataFrame:
    """係数・オッズ比・OR の信頼区間。

    区間は係数の尺度で作ってから exp する。OR の尺度で正規近似すると、下限が負に
    なりうる（オッズ比は正の量なので、それは意味を持たない）。
    """
    ci = res.conf_int(conf)
    return pd.DataFrame({
        "係数": res.b,
        "標準誤差": res.se,
        "z": res.z,
        "p": res.pvalues,
        "OR": np.exp(res.b),
        "OR_lo": np.exp(ci[:, 0]),
        "OR_hi": np.exp(ci[:, 1]),
    }, index=names or res.names)


def confusion(y, p, threshold: float = 0.5) -> pd.DataFrame:
    """混同行列。閾値を引数に出してあるのは、0.5 が既定であって正解ではないから。"""
    y = np.asarray(y, dtype=int).ravel()
    pred = (np.asarray(p, dtype=float).ravel() >= threshold).astype(int)
    table = np.array([[int(((y == a) & (pred == b)).sum()) for b in (0, 1)] for a in (0, 1)])
    return pd.DataFrame(table, index=["実際0", "実際1"], columns=["予測0", "予測1"])


def roc(y, p) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ROC 曲線の ``(fpr, tpr, thresholds)``。閾値を全部通す。"""
    y = np.asarray(y, dtype=int).ravel()
    score = np.asarray(p, dtype=float).ravel()
    order = np.argsort(-score)
    y, score = y[order], score[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    tpr = np.r_[0.0, tp / max(tp[-1], 1)]
    fpr = np.r_[0.0, fp / max(fp[-1], 1)]
    return fpr, tpr, np.r_[np.inf, score]


def auc(y, p) -> float:
    """ROC 曲線の下の面積。台形則で積む。

    「ランダムに選んだ陽性が、ランダムに選んだ陰性より高いスコアを得る確率」に等しい。
    """
    fpr, tpr, _ = roc(y, p)
    return float(np.trapezoid(tpr, fpr))


def dispersion(res: GlmResult) -> float:
    """Pearson の χ²/df。**1 を大きく超えたら過分散。**

    ポアソンは平均と分散が等しいことを仮定する。過分散はその仮定の破れで、標準誤差が
    小さく出すぎる——つまり有意になりすぎる、という形で現れる。
    """
    _, var_fn, _ = _family_parts(res.family)
    v = np.clip(var_fn(res.mu), 1e-10, None)
    return float(np.sum((res.y - res.mu) ** 2 / v) / res.df_resid)


def negbin(X, y, alpha: float | None = None, add_const: bool = True,
           names: list[str] | None = None) -> GlmResult:
    """負の二項回帰。過分散パラメータ α を推定してから重みを付け替える。

    ``alpha=None`` なら、ポアソン当てはめの残差から積率法で α を求める。分散を
    $\\mu + \\alpha\\mu^2$ とする形で、α→0 でポアソンに戻る。
    """
    pois = irls(X, y, family="poisson", add_const=add_const, names=names)
    if alpha is None:
        mu = pois.mu
        # 積率法。(y-μ)² - μ を μ² に回帰した傾きが α。
        num = np.sum((pois.y - mu) ** 2 - mu)
        alpha = float(max(num / np.sum(mu**2), 1e-8))

    Xd = pois.X
    b = pois.b.copy()
    history = [b.copy()]
    n_iter = 0
    for n_iter in range(1, 51):  # noqa: B007 — 抜けた時点の値を残す
        eta = Xd @ b
        mu = np.exp(np.clip(eta, -700, 700))
        v = mu + alpha * mu**2
        w = mu**2 / v
        z = eta + (pois.y - mu) / mu
        b_new = np.linalg.solve(Xd.T @ (Xd * w[:, None]), Xd.T @ (w * z))
        history.append(b_new.copy())
        if np.max(np.abs(b_new - b)) < 1e-9:
            b = b_new
            break
        b = b_new

    eta = Xd @ b
    mu = np.exp(np.clip(eta, -700, 700))
    v = mu + alpha * mu**2
    w = mu**2 / v
    cov = np.linalg.inv(Xd.T @ (Xd * w[:, None]))
    se = np.sqrt(np.diag(cov))
    zstat = b / se
    dev = 2 * np.sum(pois.y * np.log(np.where(pois.y > 0, pois.y / mu, 1))
                     - (pois.y + 1 / alpha) * np.log((1 + alpha * pois.y) / (1 + alpha * mu)))
    return GlmResult(b=b, se=se, z=zstat, pvalues=2 * stats.norm.sf(np.abs(zstat)),
                     deviance=float(dev), n_iter=n_iter, family="negbin",
                     X=Xd, y=pois.y, mu=mu, names=pois.names, history=history)
