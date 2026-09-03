"""検定の手書き実装。統計量を組み立てるところまで自分で書く。

``scipy.stats`` を呼べば1行で済む。それでもここに並べているのは、**検定が「仮定 →
統計量 → 分布 → 尾側確率」という4段の組み立てでしかない**ことを、コードの形で見せる
ため。呼ぶ前に一度は自分で書け、が本書の姿勢である。実務では scipy を使ってよい。

全ての関数が :class:`TestResult` を返し、``assumptions`` に仮定を文字列で持つ。
検定の結果を印字したら仮定も一緒に出る、という形にしてある。「破れ目」を機械的に
列挙できるようにするための設計で、p 値だけを取り出して仮定を捨てる書き方をしにくくする。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats

__all__ = [
    "AnovaResult", "TestResult", "adjust_pvalues", "bartlett", "binom_test", "chi2_gof",
    "chi2_independence", "cohens_d", "f_oneway", "fisher_exact", "fisher_z_ci",
    "hedges_g", "levene", "mann_whitney_u", "odds_ratio", "pearson_test", "prop_2samp",
    "spearman_test", "t_1samp", "t_ind", "t_paired", "tukey_hsd", "welch_anova",
    "wilcoxon_signed_rank",
]


@dataclass(frozen=True)
class TestResult:
    """検定の結果と、それが立っている仮定。

    ``assumptions`` を必ず持たせるのは、p 値だけを見て仮定を忘れることを、型の形で
    しにくくするため。印字すると仮定も出る。
    """

    stat: float
    pvalue: float
    df: float | None = None
    alternative: str = "two-sided"
    assumptions: list[str] = field(default_factory=list)
    name: str = ""

    def __str__(self) -> str:
        df = f", df={self.df:.4g}" if self.df is not None else ""
        head = f"{self.name}: stat={self.stat:.4f}{df}, p={self.pvalue:.4g}"
        if not self.assumptions:
            return head
        return head + "\n  仮定: " + "\n        ".join(self.assumptions)


def _tail(stat: float, dist, alternative: str) -> float:
    """統計量を尾側確率に変える。検定の最後の1段はどれも同じ形をしている。"""
    if alternative == "two-sided":
        return float(2 * min(dist.cdf(stat), dist.sf(stat)))
    if alternative == "greater":
        return float(dist.sf(stat))
    if alternative == "less":
        return float(dist.cdf(stat))
    raise ValueError(f"unknown alternative: {alternative}")


# ---------------------------------------------------------------------------
# t 検定
# ---------------------------------------------------------------------------


def t_1samp(x, mu0: float = 0.0, alternative: str = "two-sided") -> TestResult:
    """1標本 t 検定。"""
    a = np.asarray(x, dtype=float).ravel()
    n = a.size
    se = a.std(ddof=1) / np.sqrt(n)
    t = (a.mean() - mu0) / se
    return TestResult(
        stat=float(t), pvalue=_tail(t, stats.t(df=n - 1), alternative),
        df=n - 1, alternative=alternative, name="1標本t検定",
        assumptions=["観測は互いに独立", "母集団が正規（n が小さいときに効く）"],
    )


def t_ind(a, b, equal_var: bool = False, alternative: str = "two-sided") -> TestResult:
    """2標本 t 検定。**既定は Welch**（scipy の既定と逆）。

    等分散を仮定する理由が実務にほとんど無いため、既定を安全側に置いた。等分散が
    本当に成り立つときの Welch の損は小さく、成り立たないときの Student の害は大きい。
    """
    x, y = np.asarray(a, float).ravel(), np.asarray(b, float).ravel()
    n1, n2 = x.size, y.size
    v1, v2 = x.var(ddof=1), y.var(ddof=1)

    if equal_var:
        sp2 = ((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2)
        se = np.sqrt(sp2 * (1 / n1 + 1 / n2))
        df = n1 + n2 - 2
        assumptions = ["観測は互いに独立", "両群の母集団が正規", "**両群の分散が等しい**"]
        name = "Studentのt検定"
    else:
        se = np.sqrt(v1 / n1 + v2 / n2)
        # Welch–Satterthwaite。自由度が整数にならないのがこの近似の顔。
        df = (v1 / n1 + v2 / n2) ** 2 / ((v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1))
        assumptions = ["観測は互いに独立", "両群の母集団が正規（分散は等しくなくてよい）"]
        name = "Welchのt検定"

    t = (x.mean() - y.mean()) / se
    return TestResult(stat=float(t), pvalue=_tail(t, stats.t(df=df), alternative),
                      df=float(df), alternative=alternative, name=name,
                      assumptions=assumptions)


def t_paired(a, b, alternative: str = "two-sided") -> TestResult:
    """対応のある t 検定。差を取って1標本 t にするだけ。

    「対応を使う」とは、個体差を差分で消すこと。だから独立2標本より検出力が高くなる。
    """
    d = np.asarray(a, float).ravel() - np.asarray(b, float).ravel()
    res = t_1samp(d, 0.0, alternative)
    return TestResult(stat=res.stat, pvalue=res.pvalue, df=res.df, alternative=alternative,
                      name="対応のあるt検定",
                      assumptions=["ペアは互いに独立", "**差**の母集団が正規（各群ではない）"])


# ---------------------------------------------------------------------------
# ノンパラメトリック
# ---------------------------------------------------------------------------


def mann_whitney_u(a, b, alternative: str = "two-sided") -> TestResult:
    """Mann–Whitney の U 検定。順位に直してから足す。

    「中央値の差の検定」と紹介されがちだが、実際に見ているのは
    P(X > Y) が 1/2 かどうかである。分布の形が違えば、中央値が同じでも棄却されうる。
    """
    x, y = np.asarray(a, float).ravel(), np.asarray(b, float).ravel()
    n1, n2 = x.size, y.size
    ranks = stats.rankdata(np.concatenate([x, y]))
    r1 = ranks[:n1].sum()
    u1 = r1 - n1 * (n1 + 1) / 2
    u = min(u1, n1 * n2 - u1)
    mu = n1 * n2 / 2
    # 同順位の補正込みの分散
    _, counts = np.unique(np.concatenate([x, y]), return_counts=True)
    tie = np.sum(counts**3 - counts)
    n = n1 + n2
    sigma = np.sqrt(n1 * n2 / 12 * ((n + 1) - tie / (n * (n - 1))))
    z = (u - mu) / sigma
    p = 2 * stats.norm.cdf(z) if alternative == "two-sided" else (
        stats.norm.sf((u1 - mu) / sigma) if alternative == "greater"
        else stats.norm.cdf((u1 - mu) / sigma))
    return TestResult(stat=float(u1), pvalue=float(min(p, 1.0)), alternative=alternative,
                      name="Mann-WhitneyのU検定",
                      assumptions=["観測は互いに独立",
                                   "正規性は要らないが、**分布の形が同じ**でないと"
                                   "「位置のずれ」とは読めない"])


def wilcoxon_signed_rank(a, b=None, alternative: str = "two-sided") -> TestResult:
    """Wilcoxon 符号順位検定。差の絶対値に順位をつけ、符号ごとに足す。"""
    d = np.asarray(a, float).ravel() if b is None else (
        np.asarray(a, float).ravel() - np.asarray(b, float).ravel())
    d = d[d != 0]  # 差が0のペアは落とす（この扱い自体が流儀の分かれ目）
    n = d.size
    ranks = stats.rankdata(np.abs(d))
    w_plus = ranks[d > 0].sum()
    mu = n * (n + 1) / 4
    sigma = np.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    z = (w_plus - mu) / sigma
    return TestResult(stat=float(w_plus), pvalue=_tail(z, stats.norm(), alternative),
                      alternative=alternative, name="Wilcoxon符号順位検定",
                      assumptions=["ペアは互いに独立", "差の分布が0のまわりで対称"])


# ---------------------------------------------------------------------------
# 比率とカテゴリ
# ---------------------------------------------------------------------------


def binom_test(k: int, n: int, p: float = 0.5, alternative: str = "two-sided") -> TestResult:
    """二項検定。近似せず、二項分布そのもので数える。"""
    res = stats.binomtest(k, n, p, alternative=alternative)
    return TestResult(stat=float(k / n), pvalue=float(res.pvalue), alternative=alternative,
                      name="二項検定", assumptions=["試行は独立", "各試行の成功確率が一定"])


def prop_2samp(k1: int, n1: int, k2: int, n2: int, method: str = "score",
               alternative: str = "two-sided") -> TestResult:
    """2標本の比率の差。``score`` はプールした分散を使う（帰無のもとで正しい）。

    ``wald`` は各群の推定値から分散を作る。帰無仮説「2つの比率は等しい」のもとでは
    プールするほうが筋が通っており、小標本で差が出る。
    """
    p1, p2 = k1 / n1, k2 / n2
    if method == "score":
        p_pool = (k1 + k2) / (n1 + n2)
        se = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    elif method == "wald":
        se = np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    else:
        raise ValueError(f"unknown method: {method}")
    z = (p1 - p2) / se
    return TestResult(stat=float(z), pvalue=_tail(z, stats.norm(), alternative),
                      alternative=alternative, name=f"2標本比率の検定({method})",
                      assumptions=["両群の観測は独立",
                                   "正規近似が効く程度に期待度数が大きい（目安 各セル5以上）"])


def chi2_gof(observed, expected=None) -> TestResult:
    """適合度のカイ二乗検定。"""
    o = np.asarray(observed, dtype=float)
    e = np.full_like(o, o.sum() / o.size) if expected is None else np.asarray(expected, float)
    stat = float(((o - e) ** 2 / e).sum())
    df = o.size - 1
    return TestResult(stat=stat, pvalue=float(stats.chi2.sf(stat, df)), df=df,
                      name="適合度のカイ二乗検定",
                      assumptions=["観測は独立", "期待度数が各セル5以上（近似の前提）"])


def chi2_independence(table, correction: bool = False) -> TestResult:
    """独立性のカイ二乗検定。``correction`` は Yates の連続修正。

    2×2 で修正を入れるかどうかは流儀が割れる。既定を False にしてあるのは、修正が
    保守的すぎるという批判が強いため。どちらを使ったかを書けるように引数に出してある。
    """
    o = np.asarray(table, dtype=float)
    n = o.sum()
    e = np.outer(o.sum(axis=1), o.sum(axis=0)) / n
    d = np.abs(o - e)
    if correction and o.shape == (2, 2):
        d = np.maximum(d - 0.5, 0)
    stat = float((d**2 / e).sum())
    df = (o.shape[0] - 1) * (o.shape[1] - 1)
    return TestResult(stat=stat, pvalue=float(stats.chi2.sf(stat, df)), df=df,
                      name="独立性のカイ二乗検定",
                      assumptions=["観測は独立", "期待度数が各セル5以上",
                                   f"連続修正: {'あり' if correction else 'なし'}"])


def fisher_exact(table, alternative: str = "two-sided") -> TestResult:
    """Fisher の正確検定。周辺度数を固定して超幾何分布で数える。"""
    odds, p = stats.fisher_exact(np.asarray(table), alternative=alternative)
    return TestResult(stat=float(odds), pvalue=float(p), alternative=alternative,
                      name="Fisherの正確検定",
                      assumptions=["観測は独立", "**周辺度数が固定**という条件付きの枠組み"])


# ---------------------------------------------------------------------------
# 分散の等しさ
# ---------------------------------------------------------------------------


def levene(*groups, center: str = "median") -> TestResult:
    """Levene 検定。中央値からの絶対偏差に一元配置分散分析をかける。

    ``center="median"`` は Brown–Forsythe 版で、正規から外れたときに頑健。
    """
    dev = []
    for g in groups:
        a = np.asarray(g, float).ravel()
        c = np.median(a) if center == "median" else a.mean()
        dev.append(np.abs(a - c))
    res = f_oneway(*dev)
    return TestResult(stat=res.stat, pvalue=res.pvalue, df=res.df, name="Levene検定",
                      assumptions=["観測は独立", "正規性には比較的頑健"])


def bartlett(*groups) -> TestResult:
    """Bartlett 検定。正規性に強く依存する（外れると分散の差でなく非正規を拾う）。"""
    ks = [np.asarray(g, float).ravel() for g in groups]
    k, n = len(ks), sum(a.size for a in ks)
    sp2 = sum((a.size - 1) * a.var(ddof=1) for a in ks) / (n - k)
    num = (n - k) * np.log(sp2) - sum((a.size - 1) * np.log(a.var(ddof=1)) for a in ks)
    c = 1 + (sum(1 / (a.size - 1) for a in ks) - 1 / (n - k)) / (3 * (k - 1))
    stat = float(num / c)
    return TestResult(stat=stat, pvalue=float(stats.chi2.sf(stat, k - 1)), df=k - 1,
                      name="Bartlett検定",
                      assumptions=["観測は独立", "**正規性に強く依存**（破れると誤警報）"])


# ---------------------------------------------------------------------------
# 分散分析
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AnovaResult:
    """平方和の分解つきの分散分析結果。"""

    stat: float
    pvalue: float
    df: tuple[float, float]
    ss_between: float
    ss_within: float
    ss_total: float
    assumptions: list[str] = field(default_factory=list)
    name: str = "一元配置分散分析"

    @property
    def eta_squared(self) -> float:
        """効果量。群間平方和が全体の何割を説明したか。"""
        return self.ss_between / self.ss_total

    def __str__(self) -> str:
        return (f"{self.name}: F={self.stat:.4f}, df=({self.df[0]:.4g}, {self.df[1]:.4g}), "
                f"p={self.pvalue:.4g}, η²={self.eta_squared:.4f}"
                + ("\n  仮定: " + "\n        ".join(self.assumptions) if self.assumptions else ""))


def f_oneway(*groups) -> AnovaResult:
    """一元配置分散分析。平方和の分解をそのまま返す。

    F 検定は「群間のばらつき ÷ 群内のばらつき」でしかない。分解を返すのは、その比が
    どこから来たのかを読者が追えるようにするため。
    """
    ks = [np.asarray(g, float).ravel() for g in groups]
    k = len(ks)
    n = sum(a.size for a in ks)
    grand = np.concatenate(ks).mean()
    ss_b = sum(a.size * (a.mean() - grand) ** 2 for a in ks)
    ss_w = sum(((a - a.mean()) ** 2).sum() for a in ks)
    df_b, df_w = k - 1, n - k
    f = (ss_b / df_b) / (ss_w / df_w)
    return AnovaResult(stat=float(f), pvalue=float(stats.f.sf(f, df_b, df_w)),
                       df=(df_b, df_w), ss_between=float(ss_b), ss_within=float(ss_w),
                       ss_total=float(ss_b + ss_w),
                       assumptions=["観測は独立", "各群が正規", "**全群の分散が等しい**"])


def welch_anova(*groups) -> AnovaResult:
    """Welch の分散分析。等分散を仮定しない版。"""
    ks = [np.asarray(g, float).ravel() for g in groups]
    k = len(ks)
    w = np.array([a.size / a.var(ddof=1) for a in ks])
    m = np.array([a.mean() for a in ks])
    mw = (w * m).sum() / w.sum()
    num = ((w * (m - mw) ** 2).sum()) / (k - 1)
    lam = np.array([(1 - wi / w.sum()) ** 2 / (a.size - 1) for wi, a in zip(w, ks, strict=True)])
    denom = 1 + 2 * (k - 2) / (k**2 - 1) * lam.sum()
    f = num / denom
    df2 = (k**2 - 1) / (3 * lam.sum())
    # 平方和は等分散版の分解を参考値として添える
    ref = f_oneway(*groups)
    return AnovaResult(stat=float(f), pvalue=float(stats.f.sf(f, k - 1, df2)),
                       df=(k - 1, float(df2)), ss_between=ref.ss_between,
                       ss_within=ref.ss_within, ss_total=ref.ss_total,
                       name="Welchの分散分析",
                       assumptions=["観測は独立", "各群が正規（分散は等しくなくてよい）"])


def tukey_hsd(y, g, alpha: float = 0.05):
    """Tukey の HSD。全ペアを比べつつ、族全体の誤り率を α に抑える。

    ペアごとに t 検定を繰り返すと、比較の数だけ誤警報が増える。studentized range 分布を
    使うのは、その増加をあらかじめ織り込むためである。
    """
    import pandas as pd

    y = np.asarray(y, float).ravel()
    g = np.asarray(g).ravel()
    levels = np.unique(g)
    k, n = levels.size, y.size
    means = {lv: y[g == lv].mean() for lv in levels}
    sizes = {lv: int((g == lv).sum()) for lv in levels}
    mse = sum(((y[g == lv] - means[lv]) ** 2).sum() for lv in levels) / (n - k)
    q = stats.studentized_range.ppf(1 - alpha, k, n - k)

    rows = []
    for i, a in enumerate(levels):
        for b in levels[i + 1:]:
            diff = means[a] - means[b]
            se = np.sqrt(mse / 2 * (1 / sizes[a] + 1 / sizes[b]))
            half = q * se
            p = float(stats.studentized_range.sf(abs(diff) / se, k, n - k))
            rows.append({"group1": a, "group2": b, "diff": diff,
                         "lo": diff - half, "hi": diff + half,
                         "p_adj": p, "reject": p < alpha})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 相関
# ---------------------------------------------------------------------------


def pearson_test(x, y, alternative: str = "two-sided") -> TestResult:
    """Pearson の相関係数の検定。線形関係しか見ていない。"""
    a, b = np.asarray(x, float).ravel(), np.asarray(y, float).ravel()
    n = a.size
    r = float(np.corrcoef(a, b)[0, 1])
    t = r * np.sqrt((n - 2) / (1 - r**2))
    return TestResult(stat=r, pvalue=_tail(t, stats.t(df=n - 2), alternative), df=n - 2,
                      alternative=alternative, name="Pearsonの相関",
                      assumptions=["観測は独立", "2変量正規",
                                   "**線形関係しか測らない**（曲線関係は r≈0 になりうる）"])


def spearman_test(x, y, alternative: str = "two-sided") -> TestResult:
    """Spearman の順位相関。順位に直してから Pearson を取る。"""
    a = stats.rankdata(np.asarray(x, float).ravel())
    b = stats.rankdata(np.asarray(y, float).ravel())
    res = pearson_test(a, b, alternative)
    return TestResult(stat=res.stat, pvalue=res.pvalue, df=res.df, alternative=alternative,
                      name="Spearmanの順位相関",
                      assumptions=["観測は独立", "単調関係を測る（線形でなくてよい）"])


def fisher_z_ci(r: float, n: int, conf: float = 0.95) -> tuple[float, float]:
    """相関係数の信頼区間。z 変換して正規に近づけ、戻す。

    r は −1 と 1 で頭打ちになるので、素朴な正規近似では区間が範囲外へ出る。
    ``arctanh`` で無限区間へ写してから作り、``tanh`` で戻すのがこの手続き。
    """
    z = np.arctanh(r)
    half = stats.norm.ppf(0.5 + conf / 2) / np.sqrt(n - 3)
    return float(np.tanh(z - half)), float(np.tanh(z + half))


# ---------------------------------------------------------------------------
# 多重比較の補正
# ---------------------------------------------------------------------------


def adjust_pvalues(p, method: str = "holm") -> np.ndarray:
    """p 値の補正。``bonferroni`` / ``holm`` / ``bh``。

    Bonferroni と Holm は**族全体で1つでも誤るか**（FWER）を抑える。BH は
    **棄却したうちの誤りの割合**（FDR）を抑える。抑えている量が違うので、
    「どちらが厳しいか」ではなく「何を守りたいか」で選ぶ。
    """
    p = np.asarray(p, dtype=float).ravel()
    m = p.size
    if method == "bonferroni":
        return np.minimum(p * m, 1.0)
    if method == "holm":
        order = np.argsort(p)
        adj = np.empty(m)
        running = 0.0
        for rank, i in enumerate(order):
            running = max(running, (m - rank) * p[i])
            adj[i] = min(running, 1.0)
        return adj
    if method == "bh":
        order = np.argsort(p)
        adj = np.empty(m)
        running = 1.0
        for rank in range(m - 1, -1, -1):
            i = order[rank]
            running = min(running, m / (rank + 1) * p[i])
            adj[i] = min(running, 1.0)
        return adj
    raise ValueError(f"unknown method: {method}")


# ---------------------------------------------------------------------------
# 効果量
# ---------------------------------------------------------------------------


def cohens_d(a, b) -> float:
    """Cohen の d。プールした標準偏差で標準化した平均差。"""
    x, y = np.asarray(a, float).ravel(), np.asarray(b, float).ravel()
    n1, n2 = x.size, y.size
    sp = np.sqrt(((n1 - 1) * x.var(ddof=1) + (n2 - 1) * y.var(ddof=1)) / (n1 + n2 - 2))
    return float((x.mean() - y.mean()) / sp)


def hedges_g(a, b) -> float:
    """Hedges の g。小標本での d の偏りを補正したもの。"""
    x, y = np.asarray(a, float).ravel(), np.asarray(b, float).ravel()
    df = x.size + y.size - 2
    return float(cohens_d(x, y) * (1 - 3 / (4 * df - 1)))


def odds_ratio(table) -> float:
    """2×2 のオッズ比。"""
    (a, b), (c, d) = np.asarray(table, dtype=float)
    return float((a * d) / (b * c))
