"""記述統計を、教科書の式のまま numpy で書き下したもの。

``np.var`` を呼べば済む関数がここに並んでいるのは、既定値が2箇所で食い違うからである。

**``ddof``。** numpy の既定は ``ddof=0``（母分散）、pandas の既定は ``ddof=1``（不偏）。
同じデータを numpy と pandas に渡して違う数字が出るのは、この既定の差でしかない。本書は
標本から母集団を推す話をしているので ``ddof=1`` を既定に取り、引数として表に出す。

**分位点の補間。** 「第1四分位数」は一意ではない。numpy は9通りの方法を持っていて、
小標本では方法によって値が変わる。既定の ``linear`` に決め打ちせず、``method`` を
引数に出して、選んだことを見えるようにする。

どちらも「ライブラリが黙って選んでいる」種類の分岐で、本書が最初に潰しておきたいもの。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "iqr",
    "kurtosis",
    "mad",
    "mean",
    "median",
    "mode_hist",
    "quantile",
    "sd",
    "skewness",
    "summary",
    "var",
]


def _asarray(x) -> np.ndarray:
    a = np.asarray(x, dtype=float).ravel()
    if a.size == 0:
        raise ValueError("空の配列には要約統計量が定義できない")
    return a


def mean(x) -> float:
    """相加平均。定義そのまま、総和を個数で割る。"""
    a = _asarray(x)
    return float(a.sum() / a.size)


def median(x) -> float:
    """中央値。偶数個なら中央2つの平均を取る（この選択自体が慣習である）。"""
    a = np.sort(_asarray(x))
    n = a.size
    mid = n // 2
    return float(a[mid]) if n % 2 else float((a[mid - 1] + a[mid]) / 2.0)


def mode_hist(x, bins: int = 20) -> float:
    """最頻値。連続量には最頻値が存在しないので、ヒストグラムの最頻階級の中心を返す。

    ``bins`` を変えると答えが変わる。それが連続量の最頻値の正体で、隠さずに引数に出す。
    """
    a = _asarray(x)
    counts, edges = np.histogram(a, bins=bins)
    i = int(np.argmax(counts))
    return float((edges[i] + edges[i + 1]) / 2.0)


def var(x, ddof: int = 1) -> float:
    """分散。**既定は ``ddof=1``**（numpy と逆）。

    ``ddof=1`` は「標本から母分散を推定する」ときの不偏推定量。手元の n 個そのものの
    ばらつきを言いたいだけなら ``ddof=0`` が正しい。どちらを言いたいのかで決まる。
    """
    a = _asarray(x)
    if a.size <= ddof:
        raise ValueError(f"n={a.size} では ddof={ddof} の分散が定義できない")
    return float(((a - a.mean()) ** 2).sum() / (a.size - ddof))


def sd(x, ddof: int = 1) -> float:
    """標準偏差。分散の平方根。

    注意: 分散の不偏推定量の平方根は、標準偏差の不偏推定量では**ない**。平方根は
    非線形なので、不偏性は保存されない。実務では無視できる程度のずれだが、
    「ddof=1 なら不偏」と言い切れるのは分散までである。
    """
    return float(np.sqrt(var(x, ddof=ddof)))


def quantile(x, q, method: str = "linear"):
    """分位点。``method`` は numpy の9種をそのまま通す。

    小標本では方法によって値が変わる。「四分位数」という一意の値があるわけではない、
    ということを引数の形で見せるためにここに置いてある。
    """
    return np.quantile(_asarray(x), q, method=method)


def iqr(x, method: str = "linear") -> float:
    """四分位範囲。第3四分位数 − 第1四分位数。"""
    q1, q3 = quantile(x, [0.25, 0.75], method=method)
    return float(q3 - q1)


def mad(x, scale: float = 1.4826) -> float:
    """中央絶対偏差。``scale`` は正規分布での標準偏差との換算定数。

    1.4826 は ``1 / Φ⁻¹(0.75)``。正規分布のもとで MAD が標準偏差と同じ目盛りになるよう
    揃えるための係数であって、MAD そのものの性質ではない。裾の重い分布では、この換算は
    標準偏差を過小に見積もる。**ロバストな統計量に、非ロバストな換算係数が掛かっている。**
    """
    a = _asarray(x)
    return float(scale * np.median(np.abs(a - np.median(a))))


def skewness(x, bias: bool = False) -> float:
    """歪度。``bias=False`` で標本補正を掛ける（scipy の既定は ``bias=True``）。"""
    a = _asarray(x)
    n = a.size
    m = a.mean()
    m2 = ((a - m) ** 2).mean()
    m3 = ((a - m) ** 3).mean()
    if m2 == 0:
        return 0.0
    g1 = m3 / m2**1.5
    if bias:
        return float(g1)
    if n < 3:
        raise ValueError("n < 3 では補正した歪度が定義できない")
    return float(np.sqrt(n * (n - 1)) / (n - 2) * g1)


def kurtosis(x, excess: bool = True, bias: bool = False) -> float:
    """尖度。``excess=True`` なら正規分布が 0 になる超過尖度。

    「尖度」が 3 なのか 0 なのかは、超過を取っているかどうかの違いでしかない。
    ライブラリごとに既定が違うので、引数に出しておく。
    """
    a = _asarray(x)
    n = a.size
    m = a.mean()
    m2 = ((a - m) ** 2).mean()
    m4 = ((a - m) ** 4).mean()
    if m2 == 0:
        return 0.0
    g2 = m4 / m2**2
    if not bias:
        if n < 4:
            raise ValueError("n < 4 では補正した尖度が定義できない")
        g2 = ((n + 1) * (g2 - 3) + 6) * (n - 1) / ((n - 2) * (n - 3)) + 3
    return float(g2 - 3.0) if excess else float(g2)


def summary(x, ddof: int = 1, method: str = "linear") -> pd.DataFrame:
    """本書の定型要約表。``DataFrame`` を渡せば列ごとに作る。

    平均と中央値を必ず並べて出す。この2つが離れているかどうかが、「平均で語ってよい
    データか」の最初の判定になる。
    """
    if isinstance(x, pd.DataFrame):
        numeric = x.select_dtypes("number")
        return pd.concat(
            {c: summary(numeric[c].dropna(), ddof=ddof, method=method) for c in numeric},
            names=["column"],
        ).droplevel(1)

    a = _asarray(x)
    q1, q3 = quantile(a, [0.25, 0.75], method=method)
    return pd.DataFrame(
        [{
            "n": a.size,
            "mean": mean(a),
            "median": median(a),
            "sd": sd(a, ddof=ddof),
            "iqr": float(q3 - q1),
            "mad": mad(a),
            "min": float(a.min()),
            "q1": float(q1),
            "q3": float(q3),
            "max": float(a.max()),
            "skew": skewness(a),
            "kurtosis": kurtosis(a),
        }]
    )
