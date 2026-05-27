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

/*
 * Probability type.
 *
 * Use one integer bit so that prob_t can exactly represent values near 1.
 * Do NOT use ap_ufixed<COEFF_W, 0>, because that cannot represent 1.
 */
using prob_t = ap_ufixed<COEFF_W, 1>;

#include "exp_neg_lut.hpp"

/** Convert a Boolean bit to an Ising spin: false -> -1, true -> +1. */
static coeff_t spin_value(bool b) {
    if (b) {
        return coeff_t(1);
    }

    return coeff_t(-1);
}

/** Return a if spin_bit is +1, and -a if spin_bit is -1. */
static coeff_t spin_mul(coeff_t a, bool spin_bit) {
    if (spin_bit) {
        return a;
    }

    coeff_t out = coeff_t(0) - a;
    return out;
}

/**
 * Approximate exp(-x) for x >= 0 using a fixed ROM table and linear interpolation.
 *
 * For x <= 0, return 1.
 * For x >= EXP_NEG_XMAX, return 0.
 *
 * The header exp_neg_lut.hpp must define:
 *
 *   static const int EXP_NEG_LUT_SIZE;
 *   static const coeff_t EXP_NEG_XMAX;
 *   static const prob_t EXP_NEG_LUT[EXP_NEG_LUT_SIZE + 1];
 */
static prob_t exp_neg_approx(coeff_t x) {
    if (x <= coeff_t(0)) {
        return prob_t(1);
    }

    if (x >= EXP_NEG_XMAX) {
        return prob_t(0);
    }

    /*
     * z maps x in [0, EXP_NEG_XMAX] to [0, EXP_NEG_LUT_SIZE].
     *
     * For EXP_NEG_LUT_SIZE = 1024, 11 integer bits are enough.
     * Keeping this type modest avoids the negative-shift warning that can
     * appear when converting overly wide fixed-point values to int.
     */
    ap_ufixed<26, 11> z =
        ap_ufixed<26, 11>((x * coeff_t(EXP_NEG_LUT_SIZE)) / EXP_NEG_XMAX);

    unsigned int idx = z.to_uint();

    if (idx >= EXP_NEG_LUT_SIZE) {
        return EXP_NEG_LUT[EXP_NEG_LUT_SIZE];
    }

    ap_ufixed<26, 11> idx_fixed = ap_ufixed<26, 11>(idx);
    coeff_t frac = coeff_t(z - idx_fixed);

    coeff_t y0 = coeff_t(EXP_NEG_LUT[idx]);
    coeff_t y1 = coeff_t(EXP_NEG_LUT[idx + 1]);

    coeff_t out = coeff_t(y0 + frac * coeff_t(y1 - y0));

    if (out <= coeff_t(0)) {
        return prob_t(0);
    }

    if (out >= coeff_t(1)) {
        return prob_t(1);
    }

    return prob_t(out);
}

/**
 * Metropolis-Hastings accept/reject rule.
 *
 * The input u must be a uniform random fixed-point value in [0, 1).
 *
 * Accept if:
 *
 *   Delta E <= 0
 *
 * or otherwise if:
 *
 *   u <= exp(-beta Delta E).
 */
static bool metropolis_accept(coeff_t delta_energy, coeff_t beta, prob_t u) {
    if (delta_energy <= coeff_t(0)) {
        return true;
    }

    coeff_t x = coeff_t(beta * delta_energy);
    prob_t p = exp_neg_approx(x);

    return u <= p;
}

/**
 * Local single-spin-flip delta energy:
 *
 *   Delta E_i = 2 s_i (h_i + sum_{j != i} J_ij s_j).
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
            coeff_t term = spin_mul(J[i][j], s[j]);
            field = coeff_t(field + term);
        }
    }

    coeff_t out = coeff_t(coeff_t(2) * spin_value(s[i]) * field);
    return out;
}

/**
 * Uniform-proposal delta energy:
 *
 *   Delta E = E(y) - E(s),
 *
 * where:
 *
 *   E(y) = -sum_i h_i y_i - sum_{i<j} J_ij y_i y_j.
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

    coeff_t e = coeff_t(0);

    for (int i = 0; i < SK_N; ++i) {
#pragma HLS UNROLL
        coeff_t term = spin_mul(h[i], y[i]);
        e = coeff_t(e - term);
    }

    for (int i = 0; i < SK_N; ++i) {
#pragma HLS UNROLL
        for (int j = i + 1; j < SK_N; ++j) {
#pragma HLS UNROLL
            /*
             * y_i y_j = +1 if the two bits are equal, and -1 otherwise.
             * Energy term is -J_ij y_i y_j.
             */
            if (y[i] == y[j]) {
                e = coeff_t(e - J[i][j]);
            } else {
                e = coeff_t(e + J[i][j]);
            }
        }
    }

    coeff_t out = coeff_t(e - old_energy);
    return out;
}

/**
 * Full local Metropolis-Hastings trial move.
 *
 * Computes Delta E, evaluates exp(-beta Delta E), compares against u,
 * and returns the accepted energy.
 *
 * This kernel includes the Metropolis-Hastings acceptance arithmetic.
 *
 * It does not flip s_i internally. It only reports whether the proposed flip
 * should be accepted and what the resulting energy would be.
 */
void local_metropolis_step(
    const coeff_t J[SK_N][SK_N],
    const coeff_t h[SK_N],
    const bool s[SK_N],
    int i,
    coeff_t old_energy,
    coeff_t beta,
    prob_t u,
    bool *accepted,
    coeff_t *delta_energy,
    coeff_t *new_energy
) {
#pragma HLS ARRAY_PARTITION variable=J complete dim=0
#pragma HLS ARRAY_PARTITION variable=h complete dim=1
#pragma HLS ARRAY_PARTITION variable=s complete dim=1

    coeff_t de = local_delta_energy(J, h, s, i);
    bool acc = metropolis_accept(de, beta, u);

    *accepted = acc;
    *delta_energy = de;

    if (acc) {
        *new_energy = coeff_t(old_energy + de);
    } else {
        *new_energy = old_energy;
    }
}

/**
 * Full uniform-proposal Metropolis-Hastings trial move.
 *
 * Computes Delta E = E(y)-E(s), evaluates exp(-beta Delta E),
 * compares against u, and returns the accepted energy.
 *
 * This kernel includes the Metropolis-Hastings acceptance arithmetic.
 *
 * It does not overwrite the current state s. It only reports whether the
 * proposed configuration y should be accepted and what the resulting energy
 * would be.
 */
void uniform_metropolis_step(
    const coeff_t J[SK_N][SK_N],
    const coeff_t h[SK_N],
    const bool y[SK_N],
    coeff_t old_energy,
    coeff_t beta,
    prob_t u,
    bool *accepted,
    coeff_t *delta_energy,
    coeff_t *new_energy
) {
#pragma HLS ARRAY_PARTITION variable=J complete dim=0
#pragma HLS ARRAY_PARTITION variable=h complete dim=1
#pragma HLS ARRAY_PARTITION variable=y complete dim=1

    coeff_t de = uniform_delta_energy(J, h, y, old_energy);
    bool acc = metropolis_accept(de, beta, u);

    *accepted = acc;
    *delta_energy = de;

    if (acc) {
        *new_energy = coeff_t(old_energy + de);
    } else {
        *new_energy = old_energy;
    }
}
