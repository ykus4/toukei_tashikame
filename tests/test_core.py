"""中核モジュールの性質テスト。

本書の主張が寄りかかっている性質だけを試す。数字が「たまたま今日はこう出た」ではなく
「そう出るように作ってある」ことを、機械が確かめられる形にしておく。

数値そのものの固定（本文に印刷した値の回帰テスト）は ``test_book_numbers.py`` の仕事で、
そちらは examples が揃ってから足す。
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from toukei_tashikame import datasets, describe, sim

# ---------------------------------------------------------------------------
# sim — 再現性がこの本の前提
# ---------------------------------------------------------------------------


def test_repeat_is_reproducible():
    """同じシードなら同じ結果。ここが崩れると本の数字が全部崩れる。"""
    a = sim.repeat(lambda rng: rng.normal(), trials=200, seed=7, progress=False)
    b = sim.repeat(lambda rng: rng.normal(), trials=200, seed=7, progress=False)
    np.testing.assert_array_equal(a, b)


def test_repeat_differs_across_seeds():
    a = sim.repeat(lambda rng: rng.normal(), trials=200, seed=1, progress=False)
    b = sim.repeat(lambda rng: rng.normal(), trials=200, seed=2, progress=False)
    assert not np.array_equal(a, b)


def test_repeat_prefix_is_stable_across_trials():
    """trials を増やしても、i 番目の試行は同じ乱数を見る。

    子シードを ``SeedSequence(seed).spawn(trials)`` で作っているので、途中で打ち切っても
    伸ばしても前半は変わらない。掃引の途中で trials を変えたときに曲線が飛ばない、という
    実用上の性質でもある。
    """
    short = sim.repeat(lambda rng: rng.normal(), trials=50, seed=3, progress=False)
    long = sim.repeat(lambda rng: rng.normal(), trials=200, seed=3, progress=False)
    np.testing.assert_array_equal(short, long[:50])


def test_coverage_of_t_interval_is_near_nominal():
    """t 区間の被覆は名目 95% の近くに来る。ずれはシミュレーション誤差の範囲。"""

    def one_trial(rng):
        x = rng.normal(50.0, 10.0, size=20)
        se = x.std(ddof=1) / np.sqrt(20)
        h = stats.t.ppf(0.975, 19) * se
        return x.mean() - h, x.mean() + h

    res = sim.coverage(one_trial, truth=50.0, trials=4_000, seed=0, progress=False)
    # 4σ 幅で判定する。ここを 2σ にするとテスト自体が 5% の確率で落ちる。
    assert abs(res.rate - 0.95) < 4 * res.se
    assert res.covered.sum() == pytest.approx(res.rate * 4_000)


def test_coverage_rejects_bad_interval_fn():
    with pytest.raises(ValueError):
        sim.coverage(lambda rng: rng.normal(), truth=0.0, trials=10, progress=False)


def test_type_one_error_matches_alpha():
    """帰無が真なら、棄却率は α に一致する。検定の定義そのもの。"""

    def pvalue(rng):
        a, b = rng.normal(size=30), rng.normal(size=30)
        return float(stats.ttest_ind(a, b, equal_var=False).pvalue)

    res = sim.rejection_rate(pvalue, alpha=0.05, trials=4_000, seed=0, progress=False)
    assert abs(res.rate - 0.05) < 4 * res.se


def test_se_shrinks_with_trials():
    """シミュレーション誤差は √trials で縮む。trials を4倍にすれば se は半分になる。

    わざと被覆が5割前後になる幅の区間を使う。100% に張り付く区間で測ると se が 0 になり、
    「縮んだ」かどうかを判定できない。
    """

    def one_trial(rng):
        m = rng.normal(0.0, 1.0, size=10).mean()
        return m - 0.2, m + 0.2

    small = sim.coverage(one_trial, truth=0.0, trials=1_000, seed=0, progress=False)
    large = sim.coverage(one_trial, truth=0.0, trials=4_000, seed=0, progress=False)
    assert 0.1 < small.rate < 0.9
    assert large.se == pytest.approx(small.se / 2.0, rel=0.25)


# ---------------------------------------------------------------------------
# datasets — 真値を持っていること
# ---------------------------------------------------------------------------


def test_every_generator_takes_a_seed():
    """同じシードで同じデータ。違うシードで違うデータ。"""
    assert np.array_equal(datasets.normal_sample(50, seed=1), datasets.normal_sample(50, seed=1))
    assert not np.array_equal(datasets.normal_sample(50, seed=1),
                              datasets.normal_sample(50, seed=2))


def test_ab_test_knows_the_truth():
    d = datasets.ab_test(n_a=20_000, n_b=20_000, p_a=0.03, lift=0.10, seed=0)
    assert d.p_b == pytest.approx(0.033)
    assert d.lift == pytest.approx(0.10)
    # 観測は真値の周りに来るが、一致はしない。そこが推定の出発点。
    assert d.b.mean() == pytest.approx(d.p_b, abs=0.004)


def test_observational_is_confounded():
    """素朴な平均差は ATE から系統的にずれる。第17章の出発点。"""
    o = datasets.observational(n=8_000, ate=2.0, confounding=1.5, seed=0)
    assert o.naive_diff > o.ate + 0.5


def test_income_median_is_what_was_asked_for():
    """対数正規の中央値は exp(μ)。σ を変えても中央値は動かず、平均だけが動く。"""
    tight = datasets.income(n=40_000, median=400.0, sigma=0.4, seed=0)
    wide = datasets.income(n=40_000, median=400.0, sigma=1.2, seed=0)
    assert describe.median(tight) == pytest.approx(400.0, rel=0.05)
    assert describe.median(wide) == pytest.approx(400.0, rel=0.05)
    assert describe.mean(wide) > describe.mean(tight)


def test_anscombe_summaries_agree():
    """4組で平均・分散・相関が一致する。図を見なければ違いが分からない、という主張。"""
    df = datasets.anscombe()
    stats_by_set = df.groupby("dataset").apply(
        lambda g: (g.x.mean(), g.y.mean(), g.x.var(), g.y.var(), g.x.corr(g.y)),
        include_groups=False,
    )
    first = np.array(stats_by_set.iloc[0])
    for row in stats_by_set:
        np.testing.assert_allclose(np.array(row), first, atol=0.01)


def test_datasaurus_has_thirteen_matching_sets():
    df = datasets.datasaurus("all")
    assert df["dataset"].nunique() == 13
    summary = df.groupby("dataset").apply(
        lambda g: (g.x.mean(), g.y.mean(), g.x.std(), g.y.std()), include_groups=False
    )
    first = np.array(summary.iloc[0])
    for row in summary:
        np.testing.assert_allclose(np.array(row), first, atol=0.02)


def test_heavy_tailed_cauchy_has_no_stable_mean():
    """コーシーでは標本平均が n を増やしても収束しない。第5章の破れ目。"""
    means = [describe.mean(datasets.heavy_tailed(n, "cauchy", seed=0)) for n in (100, 10_000)]
    normal_means = [describe.mean(datasets.normal_sample(n, 0.0, 1.0, seed=0))
                    for n in (100, 10_000)]
    # 正規では n を100倍にすれば平均は0に寄る。コーシーではその保証がない。
    assert abs(normal_means[1]) < abs(normal_means[0])
    assert max(abs(m) for m in means) > 0.1


# ---------------------------------------------------------------------------
# describe — 既定値の落とし穴
# ---------------------------------------------------------------------------


def test_var_default_is_unbiased_unlike_numpy():
    """本書の既定は ddof=1。numpy の既定（ddof=0）と違うことを固定する。"""
    x = np.array([1.0, 2.0, 3.0, 4.0])
    assert describe.var(x) == pytest.approx(np.var(x, ddof=1))
    assert describe.var(x) != pytest.approx(np.var(x))
    assert describe.var(x, ddof=0) == pytest.approx(np.var(x))


def test_quantile_method_changes_the_answer():
    """小標本では補間の方法で四分位数が変わる。「一意の値」ではない。"""
    x = np.arange(1.0, 11.0)
    linear = describe.quantile(x, 0.25, method="linear")
    lower = describe.quantile(x, 0.25, method="lower")
    assert linear != lower


def test_mad_scaling_matches_sd_under_normality():
    """1.4826 は正規分布のもとで標準偏差と目盛りを揃える定数。"""
    x = datasets.normal_sample(50_000, mu=0.0, sigma=3.0, seed=0)
    assert describe.mad(x) == pytest.approx(3.0, rel=0.03)
    assert describe.mad(x, scale=1.0) == pytest.approx(3.0 * 0.6745, rel=0.03)


def test_mad_survives_contamination_but_sd_does_not():
    """汚染に対して MAD は持ちこたえ、標準偏差は壊れる。第1章の主張。"""
    x = datasets.contaminated(20_000, eps=0.05, scale=10.0, seed=0)
    assert describe.sd(x) > 2.0
    assert describe.mad(x) == pytest.approx(1.0, rel=0.15)


def test_skewness_and_kurtosis_of_normal_are_near_zero():
    x = datasets.normal_sample(50_000, seed=0)
    assert abs(describe.skewness(x)) < 0.05
    assert abs(describe.kurtosis(x)) < 0.08


def test_summary_puts_mean_and_median_side_by_side():
    """平均と中央値が並んでいることが、この表の唯一の仕事。"""
    out = describe.summary(datasets.income(1_000, seed=0))
    assert {"mean", "median", "sd", "mad"} <= set(out.columns)
    assert out["mean"].iloc[0] > out["median"].iloc[0]  # 右に裾を引く
