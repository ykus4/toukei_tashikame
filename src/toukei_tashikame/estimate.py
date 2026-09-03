"""推定と区間。最尤推定・ブートストラップ・信頼区間の3系統。

区間の作り方を3つ並べて置いてあるのは、どれが「正しい」かではなく、**どれが何を仮定して
いるか**を見せるため。正規近似は n が大きいことを、t 区間は母集団が正規であることを、
ブートストラップは標本が母集団の縮図であることを仮定する。仮定が破れる場所が違うので、
壊れ方も違う。

比率の区間を3つ持っているのも同じ理由で、``ci_prop_wald`` は**小標本で壊れることを
見せるために**置いてある。実務で使ってはいけない。
"""

from __future__ import annotations

import numpy as np
from scipy import optimize, stats

__all__ = [
    "boot_ci", "bootstrap", "ci_mean_t", "ci_mean_z", "ci_prop_clopper_pearson",
    "ci_prop_wald", "ci_prop_wilson", "loglik_curve", "mle_bernoulli", "mle_normal",
    "mle_poisson", "mse_decomposition",
]


# ---------------------------------------------------------------------------
# 最尤推定
# ---------------------------------------------------------------------------


def mle_normal(x, method: str = "closed") -> tuple[float, float]:
    """正規分布の最尤推定 ``(mu_hat, sigma2_hat)``。

    閉じた式では平均と、**ddof=0 の**分散になる。最尤推定量の分散が不偏でないのは
    ここで見える。``method="numeric"`` は同じ答えを対数尤度を登って求める版で、
    「最尤推定とは山を登ることだ」を実感するために置いてある。
    """
    a = np.asarray(x, dtype=float).ravel()
    if method == "closed":
        return float(a.mean()), float(a.var(ddof=0))

    def neg_loglik(theta):
        mu, log_sigma = theta
        sigma = np.exp(log_sigma)  # 正の制約を対数で外す
        return -np.sum(stats.norm.logpdf(a, mu, sigma))

    out = optimize.minimize(neg_loglik, x0=[a.mean(), np.log(a.std() + 1e-9)],
                            method="Nelder-Mead", options={"xatol": 1e-10, "fatol": 1e-10})
    mu, log_sigma = out.x
    return float(mu), float(np.exp(log_sigma) ** 2)


def mle_bernoulli(x) -> float:
    """ベルヌーイの最尤推定。標本比率そのもの。"""
    return float(np.asarray(x, dtype=float).mean())


def mle_poisson(x) -> float:
    """ポアソンの最尤推定。標本平均そのもの。"""
    return float(np.asarray(x, dtype=float).mean())


def loglik_curve(x, dist: str, grid) -> np.ndarray:
    """対数尤度の山を描くための値。``grid`` の各点での対数尤度を返す。

    最尤推定を「式を解く」ではなく「山の頂上を探す」として見せるための道具。
    """
    a = np.asarray(x, dtype=float).ravel()
    g = np.asarray(grid, dtype=float)
    if dist == "normal":  # μ を振る（σ は最尤値に固定）
        sigma = a.std(ddof=0)
        return np.array([stats.norm.logpdf(a, m, sigma).sum() for m in g])
    if dist == "bernoulli":
        return np.array([stats.bernoulli.logpmf(a, p).sum() for p in g])
    if dist == "poisson":
        return np.array([stats.poisson.logpmf(a, lam).sum() for lam in g])
    raise ValueError(f"unknown dist: {dist}")


# ---------------------------------------------------------------------------
# ブートストラップ
# ---------------------------------------------------------------------------


def bootstrap(x, stat=np.mean, B: int = 10_000, seed: int = 0) -> np.ndarray:
    """再標本 B 個の統計量。復元抽出で n 個引き直す、をそのまま書いたもの。

    「母集団から標本を引く」を「標本から再標本を引く」で真似る、というのが唯一の発想。
    だから標本が母集団の縮図でないとき（裾が重い、n が小さい）に壊れる。
    """
    a = np.asarray(x, dtype=float).ravel()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, a.size, size=(B, a.size))
    return np.apply_along_axis(stat, 1, a[idx])


def boot_ci(x, stat=np.mean, B: int = 10_000, kind: str = "percentile",
            conf: float = 0.95, seed: int = 0) -> tuple[float, float]:
    """ブートストラップ信頼区間。``percentile`` / ``basic`` / ``bca``。

    ``percentile`` は再標本の分布をそのまま切る。``basic`` は原点まわりに折り返す。
    ``bca`` は偏りと加速度で補正する。歪んだ統計量では3つが目に見えて違う。
    """
    a = np.asarray(x, dtype=float).ravel()
    theta_hat = float(stat(a))
    boots = bootstrap(a, stat, B=B, seed=seed)
    alpha = 1.0 - conf

    if kind == "percentile":
        return tuple(np.quantile(boots, [alpha / 2, 1 - alpha / 2]))

    if kind == "basic":
        lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
        return 2 * theta_hat - hi, 2 * theta_hat - lo

    if kind == "bca":
        # z0: 再標本のうち元の推定値を下回る割合から偏りを測る
        prop = np.mean(boots < theta_hat)
        prop = min(max(prop, 1e-12), 1 - 1e-12)
        z0 = stats.norm.ppf(prop)
        # a: ジャックナイフの歪みから加速度を測る
        jack = np.array([stat(np.delete(a, i)) for i in range(a.size)])
        d = jack.mean() - jack
        denom = 6.0 * (np.sum(d**2) ** 1.5)
        acc = float(np.sum(d**3) / denom) if denom > 0 else 0.0
        z = stats.norm.ppf([alpha / 2, 1 - alpha / 2])
        adj = stats.norm.cdf(z0 + (z0 + z) / (1 - acc * (z0 + z)))
        return tuple(np.quantile(boots, adj))

    raise ValueError(f"unknown kind: {kind}")


# ---------------------------------------------------------------------------
# 平均の区間
# ---------------------------------------------------------------------------


def ci_mean_z(x, conf: float = 0.95, sigma: float | None = None) -> tuple[float, float]:
    """正規近似の区間。σ を知っているか、n が十分大きいことを仮定する。"""
    a = np.asarray(x, dtype=float).ravel()
    s = a.std(ddof=1) if sigma is None else sigma
    half = stats.norm.ppf(0.5 + conf / 2) * s / np.sqrt(a.size)
    return float(a.mean() - half), float(a.mean() + half)


def ci_mean_t(x, conf: float = 0.95) -> tuple[float, float]:
    """t 区間。σ を標本から推したぶんの不確かさを、裾の重い分布で埋め合わせる。"""
    a = np.asarray(x, dtype=float).ravel()
    half = stats.t.ppf(0.5 + conf / 2, df=a.size - 1) * a.std(ddof=1) / np.sqrt(a.size)
    return float(a.mean() - half), float(a.mean() + half)


# ---------------------------------------------------------------------------
# 比率の区間 — 3つ並べる理由は「1つ目が壊れる」ことにある
# ---------------------------------------------------------------------------


def ci_prop_wald(k: int, n: int, conf: float = 0.95) -> tuple[float, float]:
    """Wald 区間。**小標本で壊れることを見せるために置いてある。**

    k=0 なら幅0の区間 [0, 0] を返す。「95%の確率で真の比率は 0 です」と言っている
    ことになる。被覆確率も名目の 95% を大きく下回る。実務で使ってはいけない。
    """
    p = k / n
    half = stats.norm.ppf(0.5 + conf / 2) * np.sqrt(p * (1 - p) / n)
    return float(p - half), float(p + half)


def ci_prop_wilson(k: int, n: int, conf: float = 0.95) -> tuple[float, float]:
    """Wilson 区間。実務の既定はこれでよい。

    Wald が「推定値まわりに正規近似」なのに対し、こちらは「真値 p のもとで観測が
    どこまで振れるか」を解いて逆に解く。k=0 でも幅が潰れない。
    """
    z = stats.norm.ppf(0.5 + conf / 2)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return float(center - half), float(center + half)


def ci_prop_clopper_pearson(k: int, n: int, conf: float = 0.95) -> tuple[float, float]:
    """Clopper–Pearson。二項分布から厳密に作るので、被覆は必ず名目以上になる。

    「必ず以上」は保守的ということでもあって、区間は他の2つより広い。安全側に倒す
    ぶんだけ、検出力を捨てている。
    """
    alpha = 1.0 - conf
    lo = 0.0 if k == 0 else float(stats.beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(stats.beta.ppf(1 - alpha / 2, k + 1, n - k))
    return lo, hi


# ---------------------------------------------------------------------------
# バイアス・バリアンス
# ---------------------------------------------------------------------------


def mse_decomposition(estimates, truth: float) -> tuple[float, float, float]:
    """``(bias2, variance, mse)``。MSE = バイアス² + バリアンス を数字で確かめる。

    「偏っているほうがマシ」なことがあるのは、この分解の右辺が2項あるからである。
    """
    e = np.asarray(estimates, dtype=float).ravel()
    bias2 = float((e.mean() - truth) ** 2)
    variance = float(e.var(ddof=0))
    mse = float(((e - truth) ** 2).mean())
    return bias2, variance, mse
