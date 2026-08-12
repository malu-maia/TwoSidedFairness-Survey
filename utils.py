"""Shared data-loading helpers used by the PCT and JME methods."""

from scipy.sparse import coo_matrix
from sklearn.decomposition import NMF


def mapping_indexes(scores, rows_idx, cols_idx):
    """Map original user/item IDs to a contiguous range starting from 0."""
    if not (len(scores) == len(rows_idx) == len(cols_idx)):
        raise ValueError("scores, rows_idx, and cols_idx must have the same length")

    row_mapping = {id: index for index, id in enumerate(sorted(set(rows_idx)))}
    col_mapping = {id: index for index, id in enumerate(sorted(set(cols_idx)))}

    mapped_rows_idx = [row_mapping[id] for id in rows_idx]
    mapped_cols_idx = [col_mapping[id] for id in cols_idx]

    return mapped_rows_idx, mapped_cols_idx


def pivot_gen_score_matrix(scores, rows_idx, cols_idx):
    """Build a dense score matrix from sparse (row, col, value) triples."""
    return coo_matrix((scores, (rows_idx, cols_idx))).toarray().astype(float)


def pref_estimation(data):
    """Estimate a dense preference matrix from observed ratings via NMF."""
    pref_estimator = NMF()
    U = pref_estimator.fit_transform(data)
    V = pref_estimator.components_
    return U.dot(V)
