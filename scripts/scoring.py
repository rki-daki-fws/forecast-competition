import warnings
import numpy as np
import pandas as pd
import scipy.stats as st


def rmse(df, grouping=["target", "location"], dist_columns=["value_y", "value_x"]):
    # caluclate per group
    # first just calc squared error, then mean per group
    df["se"] = np.square(df[dist_columns[0]] - df[dist_columns[1]])
    m = df.groupby(grouping)["se"].mean()

    return np.sqrt(m)


def prepare_quantiles(gt, pred):
    # TODO generate list of quantiles based on alphas or vice versa
    qs = [0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85,
          0.9, 0.95, 0.975, 0.990]
    # compute quantiles per group from samples
    quantiled = pred.groupby(["location", "target"])["value"].quantile(qs)

    pvt = pd.pivot_table(quantiled.reset_index(), index=["location", "target"], columns="level_2", values="value")
    qdict = {q: pvt[q].values for q in pvt.columns}
    observations = gt.set_index(["location", "target"]).loc[pvt.index, "value"].values
    return observations, qdict


def weighted_interval_score(gt, pred):
    # example using quantile data

    # caluclate per group
    # prepare quantile data. can be moved outside function, once multiple quantile-based metrics are used
    observations, qdict = prepare_quantiles(gt, pred)

    alphas = [0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    wis = _weighted_interval_score(observations, alphas, qdict)
    return list(wis)


def _weighted_interval_score(
        observations, alphas, q_dict, weights=None, percent=False, check_consistency=True
):
    """
    Compute weighted interval scores for an array of observations and a number of different predicted intervals.

    This function implements the WIS-score (2). A dictionary with the respective (alpha/2)
    and (1-(alpha/2)) quantiles for all alpha levels given in `alphas` needs to be specified.

    This is a more efficient implementation using array operations instead of repeated calls of `interval_score`.

    Parameters
    ----------
    observations : array_like
        Ground truth observations.
    alphas : iterable
        Alpha levels for (1-alpha) intervals.
    q_dict : dict
        Dictionary with predicted quantiles for all instances in `observations`.
    weights : iterable, optional
        Corresponding weights for each interval. If `None`, `weights` is set to `alphas`, yielding the WIS^alpha-score.
    percent: bool, optional
        If `True`, score is scaled by absolute value of observations to yield a percentage error. Default is `False`.
    check_consistency: bool, optional
        If `True`, quantiles in `q_dict` are checked for consistency. Default is `True`.

    Returns
    -------
    total : array_like
        Total weighted interval scores.
    sharpness : array_like
        Sharpness component of weighted interval scores.
    calibration : array_like
        Calibration component of weighted interval scores.

    (2) Bracher, J., Ray, E. L., Gneiting, T., & Reich, N. G. (2020). Evaluating epidemic forecasts in an interval format. arXiv preprint arXiv:2005.12881.
    """
    if weights is None:
        weights = np.array(alphas) / 2

    if not all(alphas[i] <= alphas[i + 1] for i in range(len(alphas) - 1)):
        # TODO how about sorting them instead
        raise ValueError("Alpha values must be sorted in ascending order.")

    reversed_weights = list(reversed(weights))

    lower_quantiles = [q_dict.get(alpha / 2) for alpha in alphas]
    upper_quantiles = [q_dict.get(1 - (alpha / 2)) for alpha in alphas]
    if any(q is None for q in lower_quantiles) or any(
            q is None for q in upper_quantiles
    ):
        raise ValueError(
            f"Quantile dictionary does not include all necessary quantiles."
        )

    lower_quantiles = np.vstack(lower_quantiles)
    upper_quantiles = np.vstack(upper_quantiles)

    # Check for consistency
    if check_consistency and np.any((upper_quantiles - lower_quantiles) < 0):
        raise ValueError("Quantiles are not consistent.")

    lower_q_alphas = (2 / np.array(alphas)).reshape((-1, 1))
    upper_q_alphas = (2 / np.array(alphas)).reshape((-1, 1))
    lower_q_alphas[-1] = 1
    upper_q_alphas[-1] = 1

    # compute score components for all intervals
    sharpnesses = upper_quantiles - lower_quantiles

    overprediction = np.clip(lower_quantiles - observations, a_min=0, a_max=None) * lower_q_alphas
    underprediction = np.clip(observations - upper_quantiles, a_min=0, a_max=None) * upper_q_alphas

    # scale to percentage absolute error
    if percent:
        sharpnesses = sharpnesses / np.abs(observations)
        underprediction = underprediction / np.abs(observations)
        overprediction = overprediction / np.abs(observations)

    totals = sharpnesses + underprediction + overprediction

    # weigh scores
    weights = np.array(weights).reshape((-1, 1))

    sharpnesses_weighted = sharpnesses * weights
    underprediction_weighted = underprediction * weights
    overprediction_weighted = overprediction * weights
    totals_weighted = totals * weights

    # normalize and aggregate all interval scores
    num_weights = sum([1 if a < 1 else 0.5 for a in alphas])

    sharpnesses_final = np.sum(sharpnesses_weighted, axis=0) / num_weights
    underprediction_final = np.sum(underprediction_weighted, axis=0) / num_weights
    overprediction_final = np.sum(overprediction_weighted, axis=0) / num_weights
    totals_final = np.sum(totals_weighted, axis=0) / num_weights

    return totals_final, sharpnesses_final, underprediction_final, overprediction_final


def mae(df, grouping=["target", "location"], dist_columns=["value_y", "value_x"]):
    """
    mean absolute error
    """
    # caluclate per group
    df["ae"] = np.abs(df[dist_columns[0]] - df[dist_columns[1]])
    m = df.groupby(grouping)["ae"].mean()

    return m.values


# TODO check if default columns are correct
def mda(df, grouping=["target", "location"], gt_col="value_y", pred_col="value_x"):
    """
    mean directional accuracy
    """
    # calculate mean forecast value per grouping first
    #pred_mean = df.groupby(grouping)[pred_col].mean().values
    #true_sign = np.sign(df[gt_col].values)
    #pred_sign = np.sign(pred_mean)
    #return np.sum(true_sign == pred_sign) / len(true_sign)  # returns scalar, should be array!
    df["da"] = np.sign(df[pred_col]) == np.sign(df[gt_col])
    m = df.groupby(grouping)["da"].mean()
    return m.values


def confidence_interval(a, alpha):
    if len(a) > 1:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return st.t.interval(1-alpha, len(a) - 1, loc=np.nanmean(a, axis=0), scale=st.sem(a, nan_policy="omit"))
    else:
        if a.ndim == 1:
            return (np.full(a.size, np.nan), np.full(a.size, np.nan))
        else:
            return (np.full(a.shape[1], np.nan), np.full(a.shape[1], np.nan))



def coverage_probability():
    # empirical coverage probability
    # is just relative fraction of true value within CI / num_total
    # TODO compute this from samples, then aggregate?!
    """
        df["se"] = np.square(df[dist_columns[0]] - df[dist_columns[1]])
    m = df.groupby(grouping)["se"].mean()

    return np.sqrt(m)
    """
    # for each target, location compute CIs and wether or not GT within CI
    # then simply compute frequency
    # on the other hand, results still has values per target, location
    # it does not make sense to calc probability at that level

    # could just do first step: compute if within CI (different levels)
    # not really a score, so a bit weird to save next to other scores, but computing at that level makes sense

    pass


def within_PI(df, grouping=["target", "location"], gt_col="value_y", pred_col="value_x", alpha=0.05):
    """
    Calculate if GT value is within 1-alpha prediction interval

    gt is an array of n,  or n, 1   <- though contain all same values. to could be broken down into scalar!?
    pred is an array of 100,  or 100, 1
    """
    # group by target, calculate quantiles
    #lower, upper = 0 + (1 - ci) / 2, 1 - (1 - ci) / 2
    lower, upper = alpha / 2, 1 - (alpha / 2)
    if not len(df):
        return np.nan
    grouped = df.groupby(grouping)
    lq = grouped[pred_col].quantile(lower)
    uq = grouped[pred_col].quantile(upper)
    gt_val = grouped[gt_col].mean()

    return np.logical_and(lq <= gt_val, gt_val <= uq).astype(np.uint8).values

