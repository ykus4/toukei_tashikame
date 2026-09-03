"""手書き実装とライブラリの照合。

本書は「呼ぶ前に一度は自分で書け」と言う。その手書きが正しいことは、機械が
確かめられる。ここが落ちたら、本文に載っている実装のほうが間違っている。
"""

from __future__ import annotations

import numpy as np
import pytest
import statsmodels.api as sm
from scipy import stats

from toukei_tashikame import datasets, describe, estimate, glm, power, regression, testing


class TestT:
    def test_one_sample_matches_scipy(self):
        x = datasets.normal_sample(30, 50, 10, seed=0)
        mine = testing.t_1samp(x, 48.0)
        theirs = stats.ttest_1samp(x, 48.0)
        assert mine.stat == pytest.approx(theirs.statistic)
        assert mine.pvalue == pytest.approx(theirs.pvalue)

    def test_welch_is_the_default(self):
        a, b = datasets.two_groups(30, 20, delta=0.5, sd_ratio=2.0, seed=0)
        assert testing.t_ind(a, b).pvalue == pytest.approx(
            stats.ttest_ind(a, b, equal_var=False).pvalue)
        assert testing.t_ind(a, b, equal_var=True).pvalue == pytest.approx(
            stats.ttest_ind(a, b, equal_var=True).pvalue)

    def test_paired_matches_scipy(self):
        a, b = datasets.paired(40, delta=0.3, seed=0)
        assert testing.t_paired(a, b).pvalue == pytest.approx(stats.ttest_rel(a, b).pvalue)


class TestNonparametric:
    def test_mann_whitney_matches_scipy(self):
        a, b = datasets.two_groups(40, 35, delta=0.6, seed=0)
        mine = testing.mann_whitney_u(a, b)
        theirs = stats.mannwhitneyu(a, b, alternative="two-sided")
        assert mine.pvalue == pytest.approx(theirs.pvalue, rel=0.05)

    def test_wilcoxon_matches_scipy(self):
        a, b = datasets.paired(40, delta=0.4, seed=0)
        mine = testing.wilcoxon_signed_rank(a, b)
        theirs = stats.wilcoxon(a, b, correction=False, mode="approx")
        assert mine.pvalue == pytest.approx(theirs.pvalue, rel=0.05)


class TestCategoricalAndVariance:
    def test_chi2_independence_matches_scipy(self):
        table = np.array([[30.0, 70.0], [45.0, 55.0]])
        mine = testing.chi2_independence(table)
        stat, p, _, _ = stats.chi2_contingency(table, correction=False)
        assert mine.stat == pytest.approx(stat)
        assert mine.pvalue == pytest.approx(p)

    def test_levene_and_bartlett_match_scipy(self):
        a, b = datasets.two_groups(40, 40, sd_ratio=2.0, seed=0)
        assert testing.levene(a, b).pvalue == pytest.approx(
            stats.levene(a, b, center="median").pvalue, rel=1e-6)
        assert testing.bartlett(a, b).pvalue == pytest.approx(stats.bartlett(a, b).pvalue)


class TestAnovaAndCorrelation:
    def test_f_oneway_matches_scipy(self):
        y, g = datasets.anova_data((0.0, 0.4, 0.9), 25, seed=0)
        groups = [y[g == i] for i in range(3)]
        mine = testing.f_oneway(*groups)
        theirs = stats.f_oneway(*groups)
        assert mine.stat == pytest.approx(theirs.statistic)
        assert mine.ss_total == pytest.approx(mine.ss_between + mine.ss_within)

    def test_pearson_and_spearman_match_scipy(self):
        x, y = datasets.bivariate_normal(60, rho=0.6, seed=0)
        assert testing.pearson_test(x, y).pvalue == pytest.approx(stats.pearsonr(x, y).pvalue)
        assert testing.spearman_test(x, y).pvalue == pytest.approx(
            stats.spearmanr(x, y).pvalue, rel=1e-6)

    def test_fisher_z_interval_covers_truth(self):
        lo, hi = testing.fisher_z_ci(0.6, 60)
        assert lo < 0.6 < hi
        assert -1 <= lo and hi <= 1


class TestMultipleComparisons:
    @pytest.mark.parametrize("method", ["bonferroni", "holm", "bh"])
    def test_adjustment_is_monotone_and_bounded(self, method):
        p = np.array([0.001, 0.01, 0.03, 0.2, 0.7])
        adj = testing.adjust_pvalues(p, method)
        assert np.all(adj >= p - 1e-12)
        assert np.all(adj <= 1.0)
        assert np.all(np.diff(adj[np.argsort(p)]) >= -1e-12)

    def test_bh_is_less_strict_than_bonferroni(self):
        p = np.array([0.001, 0.01, 0.03, 0.2, 0.7])
        assert np.all(testing.adjust_pvalues(p, "bh") <= testing.adjust_pvalues(p, "bonferroni"))


class TestRegression:
    def test_ols_matches_statsmodels(self):
        X, y, _ = datasets.regression_data(200, b=(1.0, 2.0, -0.5), seed=0)
        mine = regression.ols(X, y, add_const=False)
        theirs = sm.OLS(y, X).fit()
        np.testing.assert_allclose(mine.b, theirs.params)
        np.testing.assert_allclose(mine.se, theirs.bse)
        np.testing.assert_allclose(mine.pvalues, theirs.pvalues)
        assert mine.r2 == pytest.approx(theirs.rsquared)
        assert mine.r2_adj == pytest.approx(theirs.rsquared_adj)

    def test_vif_flags_collinearity(self):
        X, _, _ = datasets.collinear_data(300, rho=0.99, seed=0)
        assert regression.vif(X).max() > 10

    def test_ridge_shrinks_towards_zero(self):
        X, y, _ = datasets.collinear_data(200, rho=0.99, seed=0)
        plain = regression.ols(X, y, add_const=False).b
        shrunk = regression.ridge(X, y, lam=10.0, add_const=False)
        assert np.abs(shrunk[1:]).sum() < np.abs(plain[1:]).sum()

    def test_lasso_sets_some_coefficients_to_zero(self):
        rng = np.random.default_rng(0)
        X = np.column_stack([np.ones(200), rng.normal(size=(200, 5))])
        y = X[:, 1] * 2.0 + rng.normal(size=200)  # 2列目以外は無関係
        b = regression.lasso(X, y, lam=0.3)
        assert np.sum(np.abs(b[1:]) < 1e-8) >= 2


class TestGlm:
    def test_logistic_matches_statsmodels(self):
        X, y, _ = datasets.logistic_data(500, b=(-1.0, 0.8), seed=0)
        mine = glm.irls(X, y, family="binomial", add_const=False)
        theirs = sm.GLM(y, X, family=sm.families.Binomial()).fit()
        np.testing.assert_allclose(mine.b, theirs.params, atol=1e-6)
        np.testing.assert_allclose(mine.se, theirs.bse, atol=1e-6)

    def test_poisson_matches_statsmodels(self):
        X, y, _ = datasets.count_data(500, b=(0.5, 0.3), seed=0)
        mine = glm.irls(X, y, family="poisson", add_const=False)
        theirs = sm.GLM(y, X, family=sm.families.Poisson()).fit()
        np.testing.assert_allclose(mine.b, theirs.params, atol=1e-6)

    def test_dispersion_detects_overdispersion(self):
        _, y_ok, _ = datasets.count_data(800, overdispersion=1.0, seed=0)
        X, y_over, _ = datasets.count_data(800, overdispersion=4.0, seed=0)
        assert glm.dispersion(glm.irls(X, y_ok, family="poisson", add_const=False)) < 1.5
        assert glm.dispersion(glm.irls(X, y_over, family="poisson", add_const=False)) > 2.0

    def test_expit_is_stable_at_extremes(self):
        assert glm.expit(-800.0) == pytest.approx(0.0)
        assert glm.expit(800.0) == pytest.approx(1.0)

    def test_auc_of_perfect_separation_is_one(self):
        y = np.r_[np.zeros(50), np.ones(50)]
        assert glm.auc(y, y) == pytest.approx(1.0)


class TestEstimateAndPower:
    def test_mle_normal_closed_matches_numeric(self):
        x = datasets.normal_sample(200, 3.0, 2.0, seed=0)
        closed = estimate.mle_normal(x)
        numeric = estimate.mle_normal(x, method="numeric")
        assert closed[0] == pytest.approx(numeric[0], rel=1e-4)
        assert closed[1] == pytest.approx(numeric[1], rel=1e-3)
        assert closed[1] == pytest.approx(describe.var(x, ddof=0))

    def test_wald_interval_collapses_when_wilson_does_not(self):
        """Wald が壊れることを固定する。本文の主張そのもの。"""
        assert estimate.ci_prop_wald(0, 20) == (0.0, 0.0)
        lo, hi = estimate.ci_prop_wilson(0, 20)
        assert lo == pytest.approx(0.0) and hi > 0.1

    def test_clopper_pearson_is_widest(self):
        w = estimate.ci_prop_wilson(3, 20)
        cp = estimate.ci_prop_clopper_pearson(3, 20)
        assert (cp[1] - cp[0]) > (w[1] - w[0])

    def test_mse_decomposition_adds_up(self):
        est = datasets.normal_sample(5000, 1.2, 0.7, seed=0)
        bias2, var, mse = estimate.mse_decomposition(est, truth=1.0)
        assert bias2 + var == pytest.approx(mse)

    @pytest.mark.parametrize(("d", "expected"), [(0.2, 394), (0.5, 64), (0.8, 26)])
    def test_sample_size_matches_textbook(self, d, expected):
        """教科書の定番の値と一致すること。Cohen の表がそのまま出る。"""
        assert power.n_for_power(d) == expected

    def test_power_formula_matches_simulation(self):
        analytic = power.power_ttest(64, 0.5)
        counted = power.power_sim(64, 0.5, trials=4_000, seed=0)
        assert abs(analytic - counted.rate) < 4 * counted.se

    def test_winners_curse_inflates_the_effect(self):
        """検出力が低いと、有意になった効果量は系統的に過大になる。"""
        out = power.winners_curse(20, 0.2, trials=4_000, seed=0)
        assert out["d_all"] == pytest.approx(0.2, abs=0.05)
        assert out["d_significant"] > 3 * out["d_true"]
