#!/usr/bin/env python3
"""Usage: python validate_schedule_gaps.py MODEL SCHEDULE OUTPUT [--gap-fit-file FILE].

MODEL contains already normalized h followed by upper-triangular J coefficients.
SCHEDULE stores step starts and outgoing squared overlaps; beta=16 is omitted.
Place this script beside schedule_outputs_conservative, in or below the repo.
"""
import argparse
import csv
import math
import re
import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parent.parent)]

import numpy as np
from scipy.linalg import eigvalsh
from monaqa2.data.runtime import spectral_filter_polynomial_degree
from monaqa2.data.spectral_gap import get_spectral_gap_fit_by_n
from monaqa2.mcmc.model import IsingModel
from monaqa2.mcmc.distribution import get_gibbs_distribution, calculate_warmness_parameter_alpha
from monaqa2.mcmc.proposal import create_proposal_matrix_uniform
from monaqa2.mcmc.transition import create_transition_matrix

EPS, BETA_FINAL, R = 1e-2, 16.0, 50
PROPOSALS = ("uniform", "layden")
GAMMAS = 0.25 + 0.35 * (np.arange(20) + 0.5) / 20
TIMES = 2.0 + 18.0 * (np.arange(32) + 0.5) / 32
GAP_TOL = 1e-12  # Smaller numerical gaps are retained but marked unresolved.


def load_model(model_file, n):
    """Match generate_files.py and the C++ reader, without renormalizing."""
    coefficients = np.loadtxt(model_file, ndmin=1).ravel()
    if coefficients.size != n * (n + 1) // 2 or not np.all(np.isfinite(coefficients)):
        raise ValueError(f"Expected {n * (n + 1) // 2} finite coefficients: h, then J[i,j] for i<j")
    model = IsingModel.from_coefficients(n, coefficients)
    # The TXT already stores h_rescaled and J_rescaled. Undo the constructor's
    # automatic normalization before any energies are evaluated or cached.
    model.alpha = 1.0
    model.h_rescaled = model.h.copy()
    model.J_rescaled = model.J.copy()
    return model


def mix(U, theta):
    """Same in-place X layer as the supplied spectral_gaps_trotter.py."""
    c, s = np.cos(theta), -1j * np.sin(theta)
    n = U.shape[0].bit_length() - 1
    for q in range(n):
        v = U.reshape(2**q, 2, 2**(n - q - 1), -1)
        a, b = v[:, 0].copy(), v[:, 1].copy()
        v[:, 0], v[:, 1] = c*a + s*b, s*a + c*b


def trotter_proposal(E):
    """Sequential 20 x 32 average; second order X/2-Z-X/2, r=50."""
    d = len(E)
    P = np.zeros((d, d))
    for gamma in GAMMAS:
        for t in TIMES:
            U = np.eye(d, dtype=complex)
            half = gamma * t / (2 * R)
            phase = np.exp(-1j * (1 - gamma) * t * E / R)[:, None]
            mix(U, half)
            for step in range(R):
                U *= phase
                mix(U, half if step == R - 1 else 2 * half)
            P += np.abs(U)**2
    return P / (len(GAMMAS) * len(TIMES))


def degree(gap, epsilon):
    """Round the package's continuous degree UP to an even integer."""
    return 2 * math.ceil(spectral_filter_polynomial_degree(gap, epsilon) / 2)


def leakage_bound(gap_fit, gap_actual, d):
    """Bound the CHOSEN Chebyshev filter on the actual nonstationary interval."""
    if gap_fit == 1.0:  # Limit of the gap filter is x**d.
        return float(np.exp(d * np.log1p(-gap_actual))) if gap_actual < 1 else 0.0
    a = 1 - gap_fit
    # Stable arccosh expressions; log-cosh avoids overflow for large degrees.
    denominator = d * np.arcsinh(np.sqrt(gap_fit * (2 - gap_fit)) / a)
    numerator = d * np.arcsinh(np.sqrt(max(gap_fit - gap_actual, 0.0) * (2 - gap_fit - gap_actual)) / a)
    return float(np.exp(np.logaddexp(numerator, -numerator) - np.logaddexp(denominator, -denominator)))


def main(model_file, schedule_file, output_file, gap_fit_file=None):
    if output_file.resolve() in (model_file.resolve(), schedule_file.resolve()):
        raise ValueError("The output path must differ from both input paths")
    match = re.fullmatch(r"schedule_n(\d+)_idx(\d+)\.txt", schedule_file.name)
    if match is None:
        raise ValueError("Input filename must be schedule_n<n>_idx<idx>.txt")
    n, idx = map(int, match.groups())
    if not (5 <= n <= 10 and 0 <= idx < 100):
        raise ValueError("Expected n=5,...,10 and idx=0,...,99")
    model_match = re.fullmatch(r"model_n(\d+)_idx(\d+)\.txt", model_file.name)
    if model_match is not None and tuple(map(int, model_match.groups())) != (n, idx):
        raise ValueError("Model and schedule filenames identify different instances")
    schedule = np.loadtxt(schedule_file, ndmin=2)
    if schedule.shape[1] != 3 or not np.all(np.isfinite(schedule)):
        raise ValueError("Expected three finite columns: beta overlap_squared safety_margin")
    starts, overlaps, safety = schedule.T
    if starts[0] != 0 or np.any(np.diff(starts) <= 0) or starts[-1] >= BETA_FINAL:
        raise ValueError("Expected increasing step starts from 0, with final beta=16 omitted")
    if np.any((overlaps < 0) | (overlaps > 1)) or np.any(safety <= 0):
        raise ValueError("Squared overlaps must lie in [0,1] and safety factors must be positive")
    betas = np.r_[starts[1:], BETA_FINAL]
    L = len(starts)
    epsilon = (EPS / 4) / L  # Requested allocation: NO rewind multiplier.
    model = load_model(model_file, n)
    E = model.energies_rescaled
    pi = [get_gibbs_distribution(model, beta) for beta in np.r_[0.0, betas]]
    checked_overlaps = np.array([calculate_warmness_parameter_alpha(p, q)**2 for p, q in zip(pi[:-1], pi[1:])])
    if not np.allclose(overlaps, checked_overlaps, rtol=1e-6, atol=1e-8):
        j = int(np.argmax(np.abs(overlaps - checked_overlaps)))
        raise ValueError(f"Model TXT and saved overlaps disagree at beta={starts[j]:.17g} -> {betas[j]:.17g}: saved={overlaps[j]:.17g}, recomputed={checked_overlaps[j]:.17g}. Check the model/schedule pair and beta=16 endpoint.")
    print(f"n={n} idx={idx}: TXT overlaps verified; max difference={np.max(np.abs(overlaps - checked_overlaps)):.3g}", flush=True)
    # Only the original ensemble GAP FITS use package data; no HDF5 instances
    # are loaded. An optional explicit pickle overrides the package fit source.
    fit_kwargs = {} if gap_fit_file is None else {"in_file": gap_fit_file}
    fits = {}
    for name in PROPOSALS:
        fits[name] = []
        for beta in betas:
            try:
                A, b = get_spectral_gap_fit_by_n(name, np.inf, float(beta), statistic="mean+std", n_min=5, n_max=10, **fit_kwargs)
            except FileNotFoundError as exc:
                raise FileNotFoundError("The original ensemble gap-fit data are required. Supply their pickle with --gap-fit-file PATH if the package default is unavailable.") from exc
            fits[name].append(float(A * np.exp(-max(b, 0.0) * n)))
    output_file.parent.mkdir(parents=True, exist_ok=True)
    columns = "n idx proposal beta_start beta overlap overlap_saved overlap_pass safety_margin gap_fit gap_actual epsilon_filter degree_used degree_required gap_covered leakage_bound filter_pass status".split()
    with output_file.open("w", newline="") as out:
        out.write(f"# eps={EPS} L={L} epsilon_single_filter={epsilon:.17g} beta_final={BETA_FINAL}\n")
        out.write(f"# model={model_file.resolve()} schedule={schedule_file.resolve()}; TXT coefficients used directly, without further normalization\n")
        out.write("# layden actual: r=50, X/2-Z-X/2, 20 gamma midpoints in [0.25,0.60], 32 time midpoints in [2,20]\n")
        out.write("# gap_fit: package mean+std exponential fit over n=5,...,10, Metropolis; degree_required is an instance-adapted reference\n")
        writer = csv.DictWriter(out, fieldnames=columns, delimiter="\t")
        writer.writeheader()
        for name in PROPOSALS:
            print(f"n={n} idx={idx}: building {name} proposal", flush=True)
            P = create_proposal_matrix_uniform(n) if name == "uniform" else trotter_proposal(E)
            for j, beta in enumerate(betas):
                Q = create_transition_matrix(P, model, float(beta), a=np.inf)
                eigenvalues = eigvalsh(np.sqrt(np.maximum(Q * Q.T, 0.0)))
                if abs(eigenvalues[-1] - 1.0) > 1e-8:
                    raise ValueError("Discriminant does not have the expected stationary eigenvalue")
                actual = float(1 - np.max(np.abs(eigenvalues[:-1])))
                fitted = fits[name][j]
                valid_fit = np.isfinite(fitted) and 0 < fitted <= 1
                resolved = np.isfinite(actual) and actual > GAP_TOL
                used = degree(fitted, epsilon) if valid_fit else np.nan
                required = degree(actual, epsilon) if resolved else np.nan
                leak = np.nan
                status = "ok"
                if not valid_fit:
                    status = "invalid_fit"
                elif not resolved:
                    status = "unresolved_gap"
                else:
                    leak = leakage_bound(fitted, actual, used)
                passed = int(np.isfinite(leak) and leak <= epsilon)
                values = [n, idx, name, starts[j], beta, checked_overlaps[j], overlaps[j], int(checked_overlaps[j] >= 1 / np.e), safety[j], fitted, actual, epsilon, used, required, int(status == "ok" and actual >= fitted), leak, passed, status]
                writer.writerow(dict(zip(columns, values)))
                out.flush()
                print(f"{name} beta={beta:.8g}: gap_fit={fitted:.8g} gap_actual={actual:.8g} leakage_bound={leak:.8g} filter_pass={passed} status={status}", flush=True)
            del P


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="Model TXT, as written by generate_files.py")
    parser.add_argument("schedule", type=Path, help="schedule_n<n>_idx<idx>.txt")
    parser.add_argument("output", type=Path, help="Output TSV path")
    parser.add_argument("--gap-fit-file", type=Path, help="Original ensemble spectral-gap pickle; defaults to the package's gap-fit data")
    args = parser.parse_args()
    main(args.model, args.schedule, args.output, args.gap_fit_file)
