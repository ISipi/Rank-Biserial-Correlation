from numpy import var, mean, sqrt
from pandas import Series
from scipy.stats import rankdata, mannwhitneyu


class EffectSize:
    """
        Cohen's d effect size for two groups

        Use:
            cohend(1st_group (pd.Series),
                   2nd_group (pd.Series))
    """

    # function to calculate Cohen's d for independent samples
    def cohend(self, d1: Series, d2: Series) -> float:
        # calculate the size of samples
        n1, n2 = len(d1), len(d2)
        # calculate the variance of the samples
        s1, s2 = var(d1, ddof=1), var(d2, ddof=1)
        # calculate the pooled standard deviation
        s = sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
        # calculate the means of the samples
        u1, u2 = mean(d1), mean(d2)
        # calculate and return the effect size
        return (u1 - u2) / s


class RankBiserialCorrelation:
    """


        Mann-whitney U is used for getting p-value.

        Kerby 2014:
        https://journals.sagepub.com/doi/full/10.2466/11.it.3.1


        Use:
            rank_biserial_correlation.run(
                                          your_bivariate (pd.Series),
                                          your_continuous_or_nominal (pd.Series),
                                          coef='simple'
                                          )

        The lengths of your_bivariate and your_continuous_or_nominal need to be equal.
        The function will return Mann-Whitney U test results (for U statistic and p-value) and correlation coefficient.
    """

    def rank_biserial(self, x: Series, y: Series, coef: str = 'simple') -> str:
        """
        When the sum of ranks of the higher category in a x Pandas Series is 0 before averaging minus sum of higher
        ranks after averaging, the correlation coefficient can be calculated using the U-statistic from Mann-Whitney:

                    r_pb = 1-[(2*U)/(n1*n2)],

        where n1 is the number of items in the lower category of x and n2 is the number of items in the higher category
        of x. This calculation however does not take sign of the correlation into account.
        (i.e. r_pb is always positive).


        :param x: dichotomous variable, where 1 is "higher"
        :param y: unranked variable
        :param coef: either 'glass', 'u_stat', or 'simple'. This last option uses simple difference by Kerby (2014), but
        all values should result in nearly identical coefficients.
        :return: rank biserial correlation
        """

        y = rankdata(y)
        higher = y[x == 1]
        lower = y[x == 0]

        n = len(x)
        n_p = len(higher)
        n_q = len(lower)

        if coef == 'glass':
            # Glass:
            mean_low = mean(lower)
            rbc_glass = (2/n_p)*(((n+1)/2)-mean_low)
            return f"Coefficient: {rbc_glass}"

        if coef == 'u_stat':
            # with U:
            R1 = sum(lower)
            R2 = sum(higher)
            U1 = (n_p * n_q) + ((n_q * (n_q + 1)) / 2) - R1
            U2 = (n_p * n_q) + ((n_p * (n_p + 1)) / 2) - R2
            U = max(U1, U2) if U1 <= U2 else min(U1, U2)  # force U-statistic to match with stats.mannwhitneyu output

            rbc_U = 1-((2*U)/(n_p*n_q))
            return f"Coefficient: {rbc_U}; U-statistic: {U}"

        if coef == 'simple':
            i_higher = []
            i_lower = []
            num_pairs = 0
            for i in higher:
                for j in lower:
                    if i > j:
                        i_higher.append(i)
                    elif i < j:
                        i_lower.append(i)
                    num_pairs += 1

            r = (len(i_higher)/num_pairs) - (len(i_lower)/num_pairs)

            return f"Coefficient: {r}"

    def p_value(self, x: Series, y: Series) -> float:
        """ use scipy.stats.mannwwhitneyu for calculating the p-value """
        higher = y[x == 1]
        lower = y[x == 0]
        stat_ = mannwhitneyu(lower, higher)
        print(stat_)
        return stat_.pvalue

    def run(self, in_x: Series, in_y: Series, coef: str = 'simple') -> None:
        print(f"{self.rank_biserial(in_x, in_y, coef)}; p-value: {self.p_value(in_x, in_y)}")