#include <ap_fixed.h>

#ifndef SK_N
#error "SK_N is not defined. Pass -DSK_N=<number_of_spins> in the HLS cflags."
#endif

#ifndef COEFF_W
#define COEFF_W 40
#endif

#ifndef COEFF_I
#define COEFF_I 16
#endif

using coeff_t = ap_fixed<COEFF_W, COEFF_I>;

/** Convert a Boolean bit to an Ising spin: false -> -1, true -> +1. */
static coeff_t spin_value(bool b) {
    return b ? coeff_t(1) : coeff_t(-1);
}

/**
 * Local single-spin-flip delta energy:
 * Delta E_i = 2 s_i (h_i + sum_{j != i} J_ij s_j).
 *
 * Work: O(n). Fully unrolled, this becomes a parallel reduction over n terms.
 */
coeff_t local_delta_energy(
    const coeff_t J[SK_N][SK_N],
    const coeff_t h[SK_N],
    const bool s[SK_N],
    int i
) {
#pragma HLS ARRAY_PARTITION variable=J complete dim=0
#pragma HLS ARRAY_PARTITION variable=h complete dim=1
#pragma HLS ARRAY_PARTITION variable=s complete dim=1

    coeff_t field = h[i];

    for (int j = 0; j < SK_N; ++j) {
#pragma HLS UNROLL
        if (j != i) {
            field += J[i][j] * spin_value(s[j]);
        }
    }

    return coeff_t(2) * spin_value(s[i]) * field;
}

/**
 * Uniform-proposal delta energy:
 * Delta E = E(y) - E(s),
 * E(y) = -sum_i h_i y_i - sum_{i<j} J_ij y_i y_j.
 *
 * Work: O(n^2). Fully unrolled, this becomes a very wide pair-term reduction.
 */
coeff_t uniform_delta_energy(
    const coeff_t J[SK_N][SK_N],
    const coeff_t h[SK_N],
    const bool y[SK_N],
    coeff_t old_energy
) {
#pragma HLS ARRAY_PARTITION variable=J complete dim=0
#pragma HLS ARRAY_PARTITION variable=h complete dim=1
#pragma HLS ARRAY_PARTITION variable=y complete dim=1

    coeff_t e = 0;

    for (int i = 0; i < SK_N; ++i) {
#pragma HLS UNROLL
        e -= h[i] * spin_value(y[i]);
    }

    for (int i = 0; i < SK_N; ++i) {
#pragma HLS UNROLL
        for (int j = i + 1; j < SK_N; ++j) {
#pragma HLS UNROLL
            e -= J[i][j] * spin_value(y[i]) * spin_value(y[j]);
        }
    }

    return e - old_energy;
}
