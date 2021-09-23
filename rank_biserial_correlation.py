"""
    When the sum of ranks of the higher category in x before averaging minus sum of higher ranks after averaging = 0,
    the correlation coefficient can be calculated using the U-statistic from Mann-Whitney:

                r_pb = 1-[(2*U)/(n1*n2)],

    where n1 is the number of items in the lower category of x and n2 is the number of items in the higher category of x.
    This calculation however does not take sign of the correlation into account. (i.e. r_pb is always positive).

    Cureton coefficient for bracket ties is not implemented.

    Mann-whitney U is used for getting p-value.

    Kerby 2014:
    https://journals.sagepub.com/doi/full/10.2466/11.it.3.1


    import this script as:
        import rank_biserial_correlation

    and use with:
        rank_biserial_correlation.main(your_bivariate, your_continuous_or_nominal, coef='simple')

    The lengths of your_bivariate and your_continuous_or_nominal need to be equal.
    The function will return Mann-Whitney U test results (for U statistic and p-value) and correlation coefficient.

"""

import numpy as np
from scipy import stats

dummy_x = np.random.choice(a=[0,1], size=50, p=[0.2, 0.8])
dummy_y = np.random.choice(a=np.arange(1,100), size=50)


def rank_biserial(x, y, coef='simple'):
    """


    :param x: dichotomous variable, where 1 is "higher"
    :param y: unranked variable
    :param coef: either 'glass', 'u_stat', or 'simple'. This last option uses simple difference by Kerby (2014), but
    all values should result in nearly identical coefficients.
    :return: rank biserial correlation
    """

    y = stats.rankdata(y)
    higher = y[x==1]
    lower = y[x==0]

    n = len(x)
    n_p = len(higher)
    n_q = len(lower)

    if coef == 'glass':
        # Glass:
        mean_low = np.mean(lower)
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


def p_value(x, y):
    #y = stats.rankdata(y)
    higher = y[x == 1]
    lower = y[x == 0]
    stat_ = stats.mannwhitneyu(lower, higher)
    print(stat_)
    return stat_.pvalue


def main(in_x = dummy_x, in_y = dummy_y, coef='simple'):
    print(f"{rank_biserial(in_x, in_y, coef)}; p-value: {p_value(in_x, in_y)}")

