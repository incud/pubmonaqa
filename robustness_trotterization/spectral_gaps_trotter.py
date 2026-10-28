"""Second-order Trotter robustness, beta=4, using the existing normalized instances.

    python spectral_gaps_trotter.py analyze IDX [GROUP]
    python spectral_gaps_trotter.py collide

SLURM splits by (IDX, GROUP), with three balanced groups of r values.
Each r retains its separate resumable HDF5 file.
Merged result: /k=K/n/idx -> spectral gaps ordered by RS.
"""
import sys
from contextlib import ExitStack
from pathlib import Path

import h5py
import numpy as np
from monaqa2.data.instances import load_instances

BETA = 4.0
RS = (1, 2, 5, 10, 20, 30, 40, 50, 75, 100)
GROUPS = ((100, 10), (75, 30, 5, 2), (50, 40, 20, 1))  # sum(r) = 110, 112, 111
KS = (1, 2, 3, 4)             # 2^k x 2^k midpoint grid
NS, COUNT = range(3, 11), 100
OUT = Path(__file__).resolve().parent
RUNS = OUT / "runs"
MERGED = OUT / "spectral_gaps_trotter_merged.hdf5"


def grid(k):
    m = 2**k
    q = (np.arange(m) + .5) / m
    return .25 + .35 * q, 2 + 18 * q


def mix(U, theta):
    """In-place exp(-i theta sum X) on every computational-basis column."""
    c, s = np.cos(theta), -1j * np.sin(theta)
    n = U.shape[0].bit_length() - 1
    for q in range(n):
        v = U.reshape(2**q, 2, 2**(n - q - 1), -1)
        a, b = v[:, 0].copy(), v[:, 1].copy()
        v[:, 0], v[:, 1] = c*a + s*b, s*a + c*b


def proposal(E, r, gammas, times):
    """Average |(exp(-i X/2) exp(-i Z) exp(-i X/2))^r|^2."""
    d = len(E)
    P = np.zeros((d, d))
    for gamma in gammas:
        for t in times:
            U = np.eye(d, dtype=complex)
            half = gamma*t/(2*r)
            phase = np.exp(-1j*(1-gamma)*t*E/r)[:, None]
            mix(U, half)
            for step in range(r):
                U *= phase
                mix(U, half if step == r-1 else 2*half)
            P += np.abs(U)**2
    P /= len(gammas)*len(times)
    return P


def spectral_gap(P, E):
    dE = E[:, None] - E[None, :]
    Q = P * np.exp(np.minimum(0, -BETA*dE))
    np.fill_diagonal(Q, 0)
    np.fill_diagonal(Q, 1-Q.sum(axis=0))
    eig = np.linalg.eigvalsh(np.sqrt(np.maximum(Q*Q.T, 0)))
    return float(1-np.max(np.abs(eig[:-1])))


def run_file(idx, r):
    return RUNS / f"spectral_gaps_{idx:03d}_r{r:03d}.hdf5"


def analyze(idx, group=None):
    if not 0 <= idx < COUNT or (group is not None and group not in range(len(GROUPS))):
        raise ValueError("Invalid instance index or group (must be 0, 1, or 2)")
    repetitions = RS if group is None else GROUPS[group]
    RUNS.mkdir(parents=True, exist_ok=True)

    with ExitStack() as stack:
        runs = {r: stack.enter_context(h5py.File(run_file(idx, r), "a"))
                for r in repetitions}
        for r, dest in runs.items():
            if ("beta" in dest.attrs and dest.attrs["beta"] != BETA) or (
                    "r" in dest.attrs and dest.attrs["r"] != r):
                raise ValueError(f"Incompatible cache: {run_file(idx, r)}")
            dest.attrs["beta"], dest.attrs["r"] = BETA, r
        old_path = OUT / f"spectral_gaps_trotter_{idx}.hdf5"
        old = stack.enter_context(h5py.File(old_path, "r")) if old_path.exists() else None
        merged = stack.enter_context(h5py.File(MERGED, "r")) if MERGED.exists() else None

        for n in NS:  # Load each instance once for all r in this group
            missing = []
            for r in repetitions:
                for k in KS:
                    key = f"k={k}/{n}/{idx}"
                    if key in runs[r]:
                        continue
                    j = RS.index(r)
                    if any(f is not None and key in f and np.isfinite(f[key][j])
                           for f in (old, merged)):
                        continue
                    missing.append((r, k))
            if not missing:
                continue
            E = np.asarray(load_instances(n, idx).energies_rescaled, dtype=float)
            for r, k in missing:
                g, t = grid(k)
                value = spectral_gap(proposal(E, r, g, t), E)
                dest = runs[r]
                dest.require_group(f"k={k}/{n}").create_dataset(str(idx), data=value)
                dest.flush()  # Save each completed gap, independent of other jobs
                print(f"idx={idx} n={n} k={k} r={r}: {value:.8g}", flush=True)
    print(f"Finished idx={idx}, group={group if group is not None else 'all'}", flush=True)


def collide():
    missing = 0
    with h5py.File(MERGED, "a") as out:
        out.attrs["beta"], out.attrs["rs"], out.attrs["ks"] = BETA, RS, KS
        for idx in range(COUNT):
            old_path = OUT / f"spectral_gaps_trotter_{idx}.hdf5"
            with ExitStack() as stack:
                old = stack.enter_context(h5py.File(old_path, "r")) if old_path.exists() else None
                runs = {r: stack.enter_context(h5py.File(run_file(idx, r), "r"))
                        for r in RS if run_file(idx, r).exists()}
                for n in NS:
                    for k in KS:
                        key = f"k={k}/{n}/{idx}"
                        if key not in out:
                            out.require_group(f"k={k}/{n}").create_dataset(
                                str(idx), data=np.full(len(RS), np.nan))
                        result = out[key][:]
                        if old is not None and key in old:
                            old_gaps = old[key][:]
                            result[~np.isfinite(result)] = old_gaps[~np.isfinite(result)]
                        for j, r in enumerate(RS):
                            if not np.isfinite(result[j]) and r in runs and key in runs[r]:
                                result[j] = runs[r][key][()]
                        out[key][:] = result
                        missing += np.count_nonzero(~np.isfinite(result))
            print(f"Merged idx={idx}", flush=True)
    print(f"Saved {MERGED}; missing gaps: {missing}")


if __name__ == "__main__":
    if len(sys.argv) in (3, 4) and sys.argv[1] == "analyze":
        analyze(int(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) == 4 else None)
    elif sys.argv[1:] == ["collide"]:
        collide()
    else:
        sys.exit("Usage: spectral_gaps_trotter.py analyze IDX [GROUP=0,1,2] | collide")
