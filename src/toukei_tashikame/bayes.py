"""ベイズ。刻んで掛けて正規化する、から始める。

事後分布は「事前 × 尤度 を正規化したもの」でしかない。:func:`grid_posterior` が3行で
書けるのはそのためで、MCMC が要るのは正規化定数の積分が手に負えなくなってからである。
だから本書は**グリッド近似 → 共役 → MCMC** の順に進む。難しい道具から入らない。

PyMC はこのモジュールから import しない。使うのは ``examples/ch16/`` と
``examples/ch18/`` のスクリプトだけで、依存の重さをパッケージ本体に持ち込まない。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

__all__ = [
    "BetaPosterior", "MHResult", "bayes_factor_beta", "beta_binomial",
    "credible_interval", "ess", "expected_loss", "grid_posterior",
    "metropolis_hastings", "prob_b_beats_a", "rhat", "trace_diagnostics",
]


def grid_posterior(loglik, prior, grid) -> np.ndarray:
    """事後分布をグリッドで。刻んで、掛けて、正規化する。

    ``loglik`` はグリッド上の対数尤度、``prior`` は事前密度。対数で足してから
    最大値を引くのは、指数を取る前に桁を揃えるため（引いた定数は正規化で消える）。
    """
    ll = np.asarray(loglik, dtype=float)
    pr = np.asarray(prior, dtype=float)
    g = np.asarray(grid, dtype=float)
    log_post = ll + np.log(np.clip(pr, 1e-300, None))
    post = np.exp(log_post - log_post.max())
    return post / np.trapezoid(post, g)


@dataclass(frozen=True)
class BetaPosterior:
    """ベータ事後分布。共役なので、更新は足し算で済む。"""

    a: float
    b: float
    k: int
    n: int

    @property
    def mean(self) -> float:
        return self.a / (self.a + self.b)

    @property
    def mode(self) -> float:
        if self.a > 1 and self.b > 1:
            return (self.a - 1) / (self.a + self.b - 2)
        return float("nan")

    def rvs(self, size: int, seed: int = 0) -> np.ndarray:
        return np.random.default_rng(seed).beta(self.a, self.b, size=size)

    def pdf(self, x) -> np.ndarray:
        return stats.beta.pdf(x, self.a, self.b)

    def interval(self, conf: float = 0.95) -> tuple[float, float]:
        lo, hi = stats.beta.ppf([(1 - conf) / 2, 0.5 + conf / 2], self.a, self.b)
        return float(lo), float(hi)

    def __str__(self) -> str:
        lo, hi = self.interval()
        return (f"Beta({self.a:g}, {self.b:g}) 平均={self.mean:.4f} "
                f"95%信用区間=[{lo:.4f}, {hi:.4f}]（{self.k}/{self.n}）")


def beta_binomial(k: int, n: int, a: float = 1.0, b: float = 1.0) -> BetaPosterior:
    """ベータ–二項の共役更新。成功を a に、失敗を b に足すだけ。

    事前 Beta(a, b) は「すでに a-1 回成功、b-1 回失敗を見た」と読める。だから
    Beta(1,1)（一様）は「何も見ていない」に対応する。
    """
    return BetaPosterior(a=a + k, b=b + (n - k), k=k, n=n)


def credible_interval(samples, conf: float = 0.95, kind: str = "eti") -> tuple[float, float]:
    """信用区間。``eti`` は等裾、``hdi`` は最高密度。

    歪んだ事後では2つが目に見えて違う。HDI は「区間の中の密度が外より高い」ことを
    保証するが、パラメータを変換すると HDI ではなくなる。ETI は変換に強いが、
    密度の低い領域を含みうる。どちらを使ったかは書くこと。
    """
    s = np.sort(np.asarray(samples, dtype=float).ravel())
    if kind == "eti":
        lo, hi = np.quantile(s, [(1 - conf) / 2, 0.5 + conf / 2])
        return float(lo), float(hi)
    if kind == "hdi":
        n = s.size
        width = int(np.floor(conf * n))
        if width < 1:
            raise ValueError("標本が少なすぎる")
        spans = s[width:] - s[:n - width]
        i = int(np.argmin(spans))
        return float(s[i]), float(s[i + width])
    raise ValueError(f"unknown kind: {kind}")


@dataclass(frozen=True)
class MHResult:
    """Metropolis–Hastings の出力。"""

    chain: np.ndarray
    accept_rate: float
    step: float
    n_accepted: int

    def burned(self, burn: int = 1000) -> np.ndarray:
        """助走を捨てた鎖。初期値の影響が残っている前半は使わない。"""
        return self.chain[burn:]

    def __str__(self) -> str:
        return (f"MH: n={len(self.chain):,}, 受容率={self.accept_rate:.3f}, "
                f"step={self.step:g}")


def metropolis_hastings(logpost, init, n: int = 10_000, step: float = 0.5,
                        seed: int = 0) -> MHResult:
    """Metropolis–Hastings。本文に載る20行版と同一の実装。

    今いる場所の近くに提案を1つ出し、事後密度の比で受け入れるか決める。それだけ。
    対称な提案（正規）なので、提案密度の比は1に落ちて式に現れない。

    受容率は 0.2〜0.5 あたりが目安。高すぎるのは動けていない証拠で、低すぎるのは
    跳びすぎている証拠である。どちらも ``step`` で調整する。
    """
    rng = np.random.default_rng(seed)
    x = np.atleast_1d(np.asarray(init, dtype=float))
    dim = x.size
    chain = np.empty((n, dim))
    log_p = logpost(x if dim > 1 else x[0])
    accepted = 0

    for i in range(n):
        proposal = x + rng.normal(0.0, step, size=dim)
        log_q = logpost(proposal if dim > 1 else proposal[0])
        # 対数で比べる。比を取ってから指数にすると簡単に桁が溢れる。
        if np.log(rng.random()) < log_q - log_p:
            x, log_p = proposal, log_q
            accepted += 1
        chain[i] = x

    out = chain[:, 0] if dim == 1 else chain
    return MHResult(chain=out, accept_rate=accepted / n, step=step, n_accepted=accepted)


def trace_diagnostics(chain, title: str = ""):
    """トレース / 自己相関 / 累積平均の3枚組。

    1枚では足りない。トレースは動けているか、自己相関は何個ぶん独立か、累積平均は
    収束したかを見る。3つとも同じ鎖の別の側面である。
    """
    from . import plots

    c = np.asarray(chain, dtype=float).ravel()
    fig, axes = plots.figure(3, 1, h=2.0)
    pal = plots.PALETTE

    axes[0].plot(c, color=pal["posterior"], lw=0.5)
    axes[0].set_ylabel("値")
    axes[0].set_title("① トレース — 動けているか（毛虫のように見えるのが良い）")

    lags = np.arange(1, 51)
    cc = c - c.mean()
    denom = float(cc @ cc)
    acf = [float((cc[:-k] @ cc[k:]) / denom) for k in lags]
    axes[1].vlines(lags, 0, acf, color=pal["data"], lw=1.0)
    axes[1].axhline(0, color=pal["ink2"], lw=0.6)
    axes[1].set_ylabel("自己相関")
    axes[1].set_title("② 自己相関 — 何ステップで独立になるか")

    axes[2].plot(np.cumsum(c) / np.arange(1, c.size + 1), color=pal["posterior"], lw=0.9)
    axes[2].axhline(c.mean(), color=pal["truth"], lw=1.0, ls="--", dashes=(4, 2.2))
    axes[2].set_ylabel("累積平均")
    axes[2].set_xlabel("反復")
    axes[2].set_title("③ 累積平均 — 落ち着いたか")

    if title:
        fig.suptitle(title, fontsize=8)
    fig.tight_layout()
    return fig


def rhat(chains) -> float:
    """Gelman–Rubin の R̂。鎖の間のばらつきと鎖の中のばらつきを比べる。

    1つの鎖だけ見ていても収束は分からない。別々の初期値から出た複数の鎖が同じ場所に
    集まって初めて、収束したと言える。1.01 を超えたら回し直す、が今の目安。
    """
    c = np.asarray(chains, dtype=float)
    if c.ndim != 2:
        raise ValueError("chains は (鎖の数, 反復) の2次元にすること")
    _, n = c.shape
    means = c.mean(axis=1)
    b = n * means.var(ddof=1)
    w = c.var(axis=1, ddof=1).mean()
    var_hat = (n - 1) / n * w + b / n
    return float(np.sqrt(var_hat / w))


def ess(chain) -> float:
    """有効標本数。自己相関のぶんだけ目減りした「実質の n」。

    10,000 回回しても、隣どうしが強く相関していれば独立な 200 個ぶんの情報しかない、
    ということが起きる。
    """
    c = np.asarray(chain, dtype=float).ravel()
    n = c.size
    cc = c - c.mean()
    denom = float(cc @ cc)
    if denom == 0:
        return float(n)
    total = 0.0
    for k in range(1, min(n - 1, 1000)):
        r = float((cc[:-k] @ cc[k:]) / denom)
        if r < 0.05:  # 相関が消えたところで打ち切る
            break
        total += r
    return float(n / (1 + 2 * total))


def bayes_factor_beta(k: int, n: int, a: float = 1.0, b: float = 1.0,
                      p0: float = 0.5) -> float:
    """ベイズファクター。H1（ベータ事前）と H0（p=p0 の一点）の周辺尤度の比。

    p 値と違って「帰無を支持する証拠」を表せる。ただし事前分布の選び方に敏感で、
    そこが批判の的でもある。
    """
    from scipy.special import betaln

    log_m1 = betaln(a + k, b + n - k) - betaln(a, b)
    log_m0 = k * np.log(p0) + (n - k) * np.log(1 - p0)
    return float(np.exp(log_m1 - log_m0))


def prob_b_beats_a(post_a: BetaPosterior, post_b: BetaPosterior,
                   draws: int = 100_000, seed: int = 0) -> float:
    """P(B > A)。事後から引いて数えるだけ。

    p 値が答えない問いにそのまま答えるのが、A/Bテストでベイズが好まれる理由である。
    「B のほうが良い確率は 94%」は意思決定にそのまま使える形をしている。
    """
    rng = np.random.default_rng(seed)
    a = rng.beta(post_a.a, post_a.b, size=draws)
    b = rng.beta(post_b.a, post_b.b, size=draws)
    return float((b > a).mean())


def expected_loss(post_a: BetaPosterior, post_b: BetaPosterior,
                  draws: int = 100_000, seed: int = 0) -> tuple[float, float]:
    """選び間違えたときに失う量の期待値 ``(loss_a, loss_b)``。

    「B が勝つ確率 94%」だけでは決められない。負けたときの損が小さいなら、
    確率が 80% でも切り替えてよい。期待損失は、その判断を1つの数字にする。
    閾値（たとえば 0.0001）を下回ったら打ち切る、という止め方に使う。
    """
    rng = np.random.default_rng(seed)
    a = rng.beta(post_a.a, post_a.b, size=draws)
    b = rng.beta(post_b.a, post_b.b, size=draws)
    return float(np.maximum(b - a, 0).mean()), float(np.maximum(a - b, 0).mean())
