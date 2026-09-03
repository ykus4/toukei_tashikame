"""因果推論。推定値と一緒に、検証できない仮定を必ず持ち回る。

このモジュールの関数はすべて :class:`CausalResult` を返し、``assumptions`` に
**データからは確かめられない仮定**を文字列で持つ。IPW なら「観測された共変量ですべての
交絡が説明される」、DID なら「処置がなければ2群は平行に動いたはず」、RDD なら
「カットオフの近傍で他に不連続なものは無い」。

どれも観測から検証できない。だから ``__str__`` は推定値だけでなく仮定も出す。
**推定値だけを印字して仮定を落とす書き方を、しにくくするための設計である。**
観察データからの因果推論は、常に検証不能な仮定の上に立っている。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import glm, regression

__all__ = [
    "CausalResult", "balance_table", "did", "e_value", "ipw_ate", "iv_2sls",
    "match_ate", "naive_diff", "parallel_trends_plot", "propensity_score", "rdd",
]


@dataclass(frozen=True)
class CausalResult:
    """因果効果の推定値と、それが立っている検証不能な仮定。"""

    estimate: float
    se: float
    assumptions: list[str] = field(default_factory=list)
    name: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def ci(self) -> tuple[float, float]:
        h = 1.96 * self.se
        return self.estimate - h, self.estimate + h

    def __str__(self) -> str:
        lo, hi = self.ci
        head = f"{self.name}: {self.estimate:.4f}（SE {self.se:.4f}, 95%CI [{lo:.4f}, {hi:.4f}]）"
        for k, v in self.extra.items():
            head += f"\n  {k}: {v}"
        if self.assumptions:
            head += "\n  検証できない仮定:\n    - " + "\n    - ".join(self.assumptions)
        return head


def naive_diff(y, t) -> CausalResult:
    """素朴な群間差。**間違った答えの基準線として必要。**

    観察データでこれを因果効果と呼んではいけない。だが「調整するとどれだけ動くか」を
    見るには、動く前の値が要る。
    """
    y = np.asarray(y, dtype=float).ravel()
    t = np.asarray(t, dtype=float).ravel()
    y1, y0 = y[t == 1], y[t == 0]
    est = y1.mean() - y0.mean()
    se = np.sqrt(y1.var(ddof=1) / y1.size + y0.var(ddof=1) / y0.size)
    return CausalResult(estimate=float(est), se=float(se), name="素朴な群間差",
                        assumptions=["**処置がランダムに割り付けられている**"
                                     "（観察データでは、まず成り立たない）"])


def propensity_score(X, t) -> np.ndarray:
    """傾向スコア。共変量から処置確率をロジスティック回帰で推す。"""
    res = glm.irls(np.asarray(X, dtype=float), np.asarray(t, dtype=float), family="binomial")
    return res.predict()


def ipw_ate(y, t, ps, stabilized: bool = True,
            trim: tuple[float, float] = (0.01, 0.99)) -> CausalResult:
    """逆確率重み付け。「処置されにくかったのに処置された人」を重く数える。

    ``trim`` は極端な傾向スコアを切り落とす。0 や 1 に近いスコアは重みを爆発させ、
    推定値が数人に支配される。切ったことは推定対象を変えることでもあるので、
    切った件数を ``extra`` に出す。
    """
    y = np.asarray(y, dtype=float).ravel()
    t = np.asarray(t, dtype=float).ravel()
    ps = np.asarray(ps, dtype=float).ravel()

    keep = (ps > trim[0]) & (ps < trim[1])
    n_trimmed = int((~keep).sum())
    y, t, ps = y[keep], t[keep], ps[keep]

    w = np.where(t == 1, 1.0 / ps, 1.0 / (1.0 - ps))
    if stabilized:
        # 周辺確率を掛けて重みの分散を抑える。ATE の推定量としては同じものを狙う。
        p_treat = t.mean()
        w = np.where(t == 1, p_treat / ps, (1 - p_treat) / (1 - ps))

    m1 = np.sum(w * t * y) / np.sum(w * t)
    m0 = np.sum(w * (1 - t) * y) / np.sum(w * (1 - t))
    est = m1 - m0
    # サンドイッチ風の近似。傾向スコアの推定誤差は織り込んでいない（保守的でない）。
    infl = w * (t * (y - m1) / max(t.mean(), 1e-12)
                - (1 - t) * (y - m0) / max(1 - t.mean(), 1e-12))
    se = float(np.std(infl, ddof=1) / np.sqrt(y.size))

    return CausalResult(
        estimate=float(est), se=se, name="IPW による ATE",
        extra={"重みの最大": f"{w.max():.2f}", "切り落とし": f"{n_trimmed}件",
               "有効標本": f"{y.size}件"},
        assumptions=[
            "**条件付き独立**: 観測された共変量で条件づければ、処置は結果と独立"
            "（＝未観測の交絡が無い）。データからは検証できない",
            "**正値性**: どの共変量の値でも、処置される確率が0でも1でもない",
            "傾向スコアのモデルが正しく指定されている",
        ])


def match_ate(y, t, ps, caliper: float = 0.2, replace: bool = True) -> CausalResult:
    """傾向スコアマッチング。処置群の各人に、似た対照を1人あてる。

    ``caliper`` は許容距離（傾向スコアの標準偏差の何倍まで）。相手が見つからない人は
    落ちる——つまり推定対象がこっそり変わる。落ちた件数を ``extra`` に出すのはそのため。
    """
    y = np.asarray(y, dtype=float).ravel()
    t = np.asarray(t, dtype=float).ravel()
    ps = np.asarray(ps, dtype=float).ravel()

    treated = np.where(t == 1)[0]
    control = np.where(t == 0)[0]
    width = caliper * ps.std(ddof=1)

    pairs, unmatched = [], 0
    pool = list(control)
    for i in treated:
        if not pool:
            unmatched += len(treated) - len(pairs)
            break
        d = np.abs(ps[pool] - ps[i])
        j = int(np.argmin(d))
        if d[j] > width:
            unmatched += 1
            continue
        pairs.append((i, pool[j]))
        if not replace:
            pool.pop(j)

    if not pairs:
        raise ValueError("caliper 内に相手が1人も見つからない")
    diffs = np.array([y[i] - y[j] for i, j in pairs])
    return CausalResult(
        estimate=float(diffs.mean()), se=float(diffs.std(ddof=1) / np.sqrt(diffs.size)),
        name="マッチングによる ATT",
        extra={"マッチ数": f"{len(pairs)}組", "相手が見つからず除外": f"{unmatched}人",
               "復元抽出": replace},
        assumptions=[
            "**条件付き独立**（IPW と同じ。検証できない）",
            "マッチできた処置群についての効果（ATT）であって、母集団全体の ATE ではない",
            f"caliper={caliper} を超える人は捨てている。推定対象が定義から変わっている",
        ])


def balance_table(X, t, weights=None, names: list[str] | None = None) -> pd.DataFrame:
    """標準化平均差（SMD）。調整の前後で共変量が揃ったかを見る。

    0.1 未満なら揃った、という慣習がある。**p 値で釣り合いを見てはいけない**——
    n が減れば p は大きくなるので、マッチングで標本を減らすほど「揃って見える」。
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    t = np.asarray(t, dtype=float).ravel()
    w = np.ones_like(t) if weights is None else np.asarray(weights, dtype=float).ravel()
    names = names or [f"x{i + 1}" for i in range(X.shape[1])]

    rows = []
    for j, name in enumerate(names):
        x = X[:, j]
        m1 = np.average(x[t == 1], weights=w[t == 1])
        m0 = np.average(x[t == 0], weights=w[t == 0])
        pooled = np.sqrt((x[t == 1].var(ddof=1) + x[t == 0].var(ddof=1)) / 2)
        smd = (m1 - m0) / pooled if pooled > 0 else 0.0
        rows.append({"変数": name, "処置群平均": m1, "対照群平均": m0,
                     "SMD": smd, "揃った(|SMD|<0.1)": abs(smd) < 0.1})
    return pd.DataFrame(rows).set_index("変数")


def did(df: pd.DataFrame, unit: str = "unit", time: str = "period",
        treat: str = "treated", y: str = "y") -> CausalResult:
    """差分の差分。2×2 の手計算と、双方向固定効果（TWFE）の回帰を両方返す。

    2群2期なら両者は一致する。一致しなくなるのは、処置のタイミングがばらけたとき——
    そこが近年の DID 批判の中心である。
    """
    g = df.groupby([treat, time])[y].mean()
    did_2x2 = (g[(1, 1)] - g[(1, 0)]) - (g[(0, 1)] - g[(0, 0)])

    d = df.copy()
    d["interaction"] = d[treat] * d[time]
    X = np.column_stack([np.ones(len(d)), d[treat], d[time], d["interaction"]])
    res = regression.ols(X, d[y].to_numpy(), add_const=False,
                         names=["const", treat, time, "did"])

    return CausalResult(
        estimate=float(res.b[3]), se=float(res.se[3]), name="DID（差分の差分）",
        extra={"2×2 手計算": f"{did_2x2:.4f}", "TWFE 回帰": f"{res.b[3]:.4f}"},
        assumptions=[
            "**平行トレンド**: 処置がなければ、2群は同じ傾きで動いたはず。"
            "反実仮想なので観測できない",
            "処置以外に、同じタイミングで片方の群だけに起きた変化が無い",
            "処置の効果が波及して対照群に及んでいない（SUTVA）",
        ])


def parallel_trends_plot(df: pd.DataFrame, unit: str = "unit", time: str = "period",
                         treat: str = "treated", y: str = "y", title: str = ""):
    """処置前の傾きを目で比べる図。平行トレンドの「傍証」にはなる。

    処置前が平行でも、処置後も平行だったとは言えない。だからこれは仮定の検証ではなく、
    仮定が明らかに破れていないかの確認である。
    """
    from . import plots

    fig, ax = plots.figure()
    pal = plots.PALETTE
    for value, color, label in ((0, pal["data"], "対照群"), (1, pal["estimate"], "処置群")):
        sub = df[df[treat] == value].groupby(time)[y].mean()
        ax.plot(sub.index, sub.to_numpy(), marker="o", ms=3, color=color, lw=1.2)
        ax.annotate(label, xy=(sub.index[-1], sub.to_numpy()[-1]), fontsize=6.2,
                    color=color, ha="right", va="bottom",
                    xytext=(-2, 3), textcoords="offset points")
    ax.set_xlabel(time)
    ax.set_ylabel(y)
    if title:
        ax.set_title(title)
    return fig


def rdd(x, y, cutoff: float = 0.0, bandwidth: float | None = None,
        order: int = 1) -> CausalResult:
    """回帰不連続。カットオフの両側で当てはめて、境界での段差を測る。

    ``bandwidth`` を狭めると偏りは減るが分散は増える。この綱引きが RDD の実務の
    ほとんどで、帯域を1つだけ報告するのは不誠実とされる。
    """
    x = np.asarray(x, dtype=float).ravel() - cutoff
    y = np.asarray(y, dtype=float).ravel()
    if bandwidth is not None:
        keep = np.abs(x) <= bandwidth
        x, y = x[keep], y[keep]

    right = (x >= 0).astype(float)
    cols = [np.ones(x.size), right]
    for p in range(1, order + 1):
        cols += [x**p, right * x**p]  # 両側で傾きを別にする
    X = np.column_stack(cols)
    res = regression.ols(X, y, add_const=False)

    return CausalResult(
        estimate=float(res.b[1]), se=float(res.se[1]), name="RDD（回帰不連続）",
        extra={"帯域": bandwidth if bandwidth is not None else "全データ",
               "多項式次数": order, "有効標本": f"{x.size}件"},
        assumptions=[
            "**カットオフの近傍で、他に不連続に変わるものが無い**",
            "対象者がカットオフをまたいで自分を操作できない",
            "カットオフ近傍の効果（局所平均処置効果）であって、全体の ATE ではない",
        ])


def iv_2sls(y, d, z, X=None) -> CausalResult:
    """操作変数法（2段階最小二乗）。第1段のF統計量を必ず返す。

    F < 10 は弱操作変数の目安。弱いと、2SLS の偏りは OLS の偏りより**大きくなりうる**。
    だから第1段の F を出さない 2SLS の報告は読めない。
    """
    y = np.asarray(y, dtype=float).ravel()
    d = np.asarray(d, dtype=float).ravel()
    z = np.asarray(z, dtype=float)
    if z.ndim == 1:
        z = z[:, None]
    extra_cols = [] if X is None else [np.asarray(X, dtype=float).reshape(len(y), -1)]

    # 第1段: 内生変数を操作変数で説明する
    first_X = np.column_stack([np.ones(len(y)), z, *extra_cols])
    first = regression.ols(first_X, d, add_const=False)
    # 操作変数だけを落としたモデルとの F 比較
    restricted = regression.ols(np.column_stack([np.ones(len(y)), *extra_cols])
                                if extra_cols else np.ones((len(y), 1)),
                                d, add_const=False)
    q = z.shape[1]
    f_stat = ((np.sum(restricted.resid**2) - np.sum(first.resid**2)) / q) / first.sigma2

    # 第2段: 予測値で置き換えて回帰する
    second_X = np.column_stack([np.ones(len(y)), first.fitted, *extra_cols])
    second = regression.ols(second_X, y, add_const=False)

    return CausalResult(
        estimate=float(second.b[1]), se=float(second.se[1]), name="IV（2SLS）",
        extra={"第1段のF": f"{f_stat:.2f}",
               "弱操作変数の疑い": "あり（F<10）" if f_stat < 10 else "なし"},
        assumptions=[
            "**除外制約**: 操作変数は、内生変数を通してしか結果に影響しない。"
            "検証できない",
            "操作変数が未観測の交絡と無相関",
            "単調性（誰も天邪鬼に動かない）。効果は LATE として読む",
        ])


def e_value(rr: float, lo: float | None = None) -> float:
    """E値。「この関連を説明し尽くすのに、未観測の交絡はどれだけ強い必要があるか」。

    未観測の交絡が無いことは証明できない。ならば、どれだけ強い交絡があれば結論が
    ひっくり返るかを数字にする、という開き直り方をする。
    """
    def _ev(r: float) -> float:
        r = 1.0 / r if r < 1 else r
        return float(r + np.sqrt(r * (r - 1)))

    return _ev(rr) if lo is None else (1.0 if (lo <= 1 <= rr or rr <= 1 <= lo) else _ev(lo))
