"""Portable extraction of the recorded selection and error-analysis rules.

This module does not fit models, acquire data, or choose using evaluation labels.
The error decomposition is explicitly post-hoc. Original experiment scripts are
retained separately in archive/; parity is checked against their saved outputs.
"""
from itertools import product

import numpy as np

THRESHOLDS = np.array(list(product((0.25, 0.5, 0.75, 1.01), repeat=2)))
FIXED_COEFFICIENTS = tuple(product((0.25, 0.5, 1.0, None), repeat=2))
GLOBAL_IDS = (0, 5, 10, 15)


def binary(values, name):
    values = np.asarray(values)
    if values.ndim != 1 or not np.isin(values, (0, 1)).all():
        raise ValueError(f"{name} must be a one-dimensional binary array")
    return values.astype(np.uint8)


def metrics(y, prediction, first=None):
    y, prediction = binary(y, "label"), binary(prediction, "prediction")
    if y.shape != prediction.shape or not len(y):
        raise ValueError("Labels and predictions must be nonempty and aligned")
    tn, fp, fn, tp = [int(v.sum()) for v in (
        (y == 0) & (prediction == 0), (y == 0) & (prediction == 1),
        (y == 1) & (prediction == 0), (y == 1) & (prediction == 1))]
    result = dict(n=len(y), tn=tn, fp=fp, fn=fn, tp=tp,
                  macro_f1=tp / max(2*tp+fp+fn, 1) + tn / max(2*tn+fp+fn, 1),
                  malicious_recall=tp / max(tp+fn, 1),
                  false_positive_rate=fp / max(tn+fp, 1))
    if first is not None:
        first = binary(first, "first prediction")
        if first.shape != y.shape:
            raise ValueError("First-expert predictions are not aligned")
        result.update(C=int(((first != y) & (prediction == y)).sum()),
                      D=int(((first == y) & (prediction != y)).sum()),
                      switches=int((prediction != first).sum()))
        if result["C"] + result["D"] != result["switches"]:
            raise AssertionError("Binary switching identity failed")
    return result


def validate_matrix(y, matrix):
    y = binary(y, "label")
    matrix = np.asarray(matrix)
    if matrix.ndim != 2 or matrix.shape[0] != len(y) or matrix.shape[1] == 0:
        raise ValueError("Expected a nonempty samples-by-candidates matrix")
    if not np.isin(matrix, (0, 1)).all():
        raise ValueError("Candidate predictions must be binary")
    return matrix


def choose(y, first, matrix, candidate_ids=None):
    """Selection-role labels ONLY. Remaining ties take first supplied candidate."""
    matrix = validate_matrix(y, matrix)
    ids = list(range(matrix.shape[1])) if candidate_ids is None else list(candidate_ids)
    if not ids or len(set(ids)) != len(ids) or min(ids) < 0 or max(ids) >= matrix.shape[1]:
        raise ValueError("Invalid candidate index set")
    records = [dict(candidate=j, **metrics(y, matrix[:, j], first))
               for j in range(matrix.shape[1])]
    recall = metrics(y, first)["malicious_recall"]
    feasible = [j for j in ids if records[j]["malicious_recall"] >= recall - 1e-12]
    if not feasible:
        raise ValueError("No recall-feasible candidate; do not relax the constraint")
    rank = lambda j: (records[j]["macro_f1"], records[j]["malicious_recall"],
                      -records[j]["switches"])
    selected = max(feasible, key=rank)
    optimal = [j for j in feasible if rank(j) == rank(selected)]
    return selected, feasible, optimal, records


def pattern_support(matrix):
    matrix = np.asarray(matrix)
    if matrix.ndim != 2 or min(matrix.shape) == 0:
        raise ValueError("Expected a nonempty prediction matrix")
    return dict(m=int(np.any(matrix != matrix[:, :1], axis=1).sum()),
                K=int(np.unique(matrix.T, axis=0).shape[0]))


def fixed_matrix(first_probability, second_probability, trigger, r_first, r_second):
    a, b = np.asarray(first_probability), np.asarray(second_probability)
    trigger = np.asarray(trigger)
    if a.ndim != 1 or a.shape != b.shape or a.shape != trigger.shape:
        raise ValueError("Unaligned expert inputs")
    if not (np.isfinite(a).all() and np.isfinite(b).all()
            and np.isin(trigger, (0, 1)).all()
            and np.all((a >= 0) & (a <= 1)) and np.all((b >= 0) & (b <= 1))
            and 0 <= r_first <= 1 and 0 <= r_second <= 1):
        raise ValueError("Invalid probabilities, trigger or reliability")
    first, second = a >= .5, b >= .5
    margin_first, margin_second = abs(a-.5)*r_first, abs(b-.5)*r_second
    columns = []
    for lo, hi in FIXED_COEFFICIENTS:
        switch = np.zeros(len(a), dtype=bool)
        for direction, coefficient in ((False, lo), (True, hi)):
            if coefficient is not None:
                mask = (first == direction) & trigger.astype(bool) & (first != second)
                switch[mask] = margin_second[mask] > coefficient*margin_first[mask]
        columns.append(np.where(switch, second, first).astype(np.uint8))
    return np.column_stack(columns)


def decomposition(selection_y, selection_first, selection_matrix, evaluation_y,
                  evaluation_first, evaluation_matrix):
    """Finite-set hindsight identity, NEVER a deployed selection procedure."""
    sm = validate_matrix(selection_y, selection_matrix)
    em = validate_matrix(evaluation_y, evaluation_matrix)
    if sm.shape[1] != em.shape[1]:
        raise ValueError("Candidate columns differ across roles")
    best, feasible, optimal, sr = choose(selection_y, selection_first, sm)
    equivalent = [j for j in feasible if np.array_equal(sm[:, j], sm[:, best])]
    losses = (em != np.asarray(evaluation_y)[:, None]).sum(axis=0)
    selected_loss = int(losses[best])
    within_best = int(losses[equivalent].min())
    feasible_best = int(losses[feasible].min())
    within, between = selected_loss - within_best, within_best - feasible_best
    f1best = [j for j in feasible if sr[j]["macro_f1"] == max(sr[k]["macro_f1"] for k in feasible)]
    er = [metrics(evaluation_y, em[:, j], evaluation_first) for j in equivalent]
    return dict(selected_candidate=best, feasible_candidates=len(feasible),
                best_rank_tuple_candidates=len(optimal),
                best_rank_tuple_patterns=pattern_support(sm[:, optimal])["K"],
                best_f1_candidates=len(f1best), best_f1_patterns=pattern_support(sm[:, f1best])["K"],
                selected_pattern_candidates=len(equivalent),
                selected_pattern_outer_patterns=pattern_support(em[:, equivalent])["K"],
                outer_FP_min=min(r["fp"] for r in er), outer_FP_max=max(r["fp"] for r in er),
                outer_FN_min=min(r["fn"] for r in er), outer_FN_max=max(r["fn"] for r in er),
                selected_outer_errors=selected_loss, oracle_E_errors=within_best,
                oracle_F_errors=feasible_best, within_class_error_space=within,
                between_class_error_space=between, total_error_space=selected_loss-feasible_best,
                closure_residual=(selected_loss-feasible_best)-within-between)
