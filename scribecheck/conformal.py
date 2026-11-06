"""Conformal risk control (CRC) for sentence-level review allocation.

A sentence is flagged for clinician review when its error score >= lambda.
Per-note loss = (severity-weighted) fraction of that note's erroneous sentences NOT flagged
(0 for notes without errors). CRC (Angelopoulos et al., 2022) picks the largest lambda (least
review) such that  (n * Rhat(lambda) + B) / (n + 1) <= alpha,  B = 1, which guarantees
E[loss] <= alpha on an exchangeable new note.

Stratified (Mondrian) CRC runs the same procedure separately inside observable sentence strata
(e.g. medication/plan sentences vs everything else), giving a guarantee *inside each stratum*.
"""
import numpy as np


def note_contribs(df, score_col, weight_col=None, mask=None):
    """For each erroneous sentence, its share of its note's total error weight.
    Returns (scores_of_error_sentences, contrib, n_notes)."""
    d = df if mask is None else df[mask]
    err = d[d.label == 1]
    w = err[weight_col].values if weight_col else np.ones(len(err))
    tot = err.assign(_w=w).groupby("iid")._w.transform("sum").values
    return err[score_col].values, w / tot, d.iid.nunique()


def risk_curve(err_scores, contrib, n_notes, grid):
    """Rhat(lambda) for each lambda in grid (ascending). Loss counts errors with score < lambda."""
    order = np.argsort(err_scores)
    s, c = err_scores[order], np.cumsum(contrib[order])
    k = np.searchsorted(s, grid, side="left")  # number of error sentences with score < lambda
    return np.where(k > 0, c[np.maximum(k - 1, 0)], 0.0) / max(n_notes, 1)


def crc_threshold(err_scores, contrib, n_notes, alpha, grid, B=1.0, delta=None):
    r = risk_curve(err_scores, contrib, n_notes, grid)
    if delta is None:
        ok = (n_notes * r + B) / (n_notes + 1) <= alpha
    else:  # high-probability variant: Hoeffding UCB, risk <= alpha with prob >= 1 - delta
        ok = r + np.sqrt(np.log(1 / delta) / (2 * max(n_notes, 1))) <= alpha
    if not ok.any():
        return grid[0]  # flag everything
    # risk is nondecreasing in lambda: take the largest lambda that still satisfies the bound
    return grid[np.where(ok)[0].max()]


def realized(df, score_col, lam, weight_col=None, mask=None):
    """Realized mean note loss and review workload at threshold lam on df."""
    d = df if mask is None else df[mask]
    flagged = d[score_col].values >= lam
    err = d.label.values == 1
    w = d[weight_col].values if weight_col else np.ones(len(d))
    tmp = d.assign(_m=(err & ~flagged) * w, _t=err * w)
    g = tmp.groupby("iid")[["_m", "_t"]].sum()
    loss = np.where(g._t > 0, g._m / g._t.replace(0, 1), 0.0)
    return loss.mean(), flagged.mean()


def grid_for(scores, k=1500):
    q = np.unique(np.quantile(scores, np.linspace(0, 1, k)))
    return np.concatenate([[-np.inf], q, [np.inf]])
