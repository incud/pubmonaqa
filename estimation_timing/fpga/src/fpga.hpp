#pragma once
#include <ap_fixed.h>
#include <ap_int.h>
#include "fixed_config.hpp"
#include "wallace.hpp"

constexpr int NUM_J = SK_N * (SK_N - 1) / 2;
constexpr int NUM_TERMS = SK_N + NUM_J;
constexpr int ENERGY_BITS = INT_BITS + FRAC;

// Physical coefficients and energies share a 2^-FRAC grid; there is no energy rescaling.
using energy_t = ap_fixed<ENERGY_BITS, INT_BITS, AP_RND, AP_SAT>;
using coeff_t = energy_t;
using prob_t = ap_ufixed<32, 1, AP_RND, AP_SAT>;
using poly_t = ap_fixed<48, 6, AP_RND, AP_SAT>;
#include "exp_cheby.hpp"

static ap_int<ENERGY_BITS> energy_bits(energy_t value) {
#pragma HLS INLINE
    ap_int<ENERGY_BITS> bits;
    bits.range(ENERGY_BITS - 1, 0) = value.range(ENERGY_BITS - 1, 0);
    return bits;
}

static energy_t energy_from_bits(ap_int<ENERGY_BITS> bits) {
#pragma HLS INLINE
    energy_t value;
    value.range(ENERGY_BITS - 1, 0) = bits.range(ENERGY_BITS - 1, 0);
    return value;
}

static energy_t energy_add(energy_t a, energy_t b, bool subtract = false) {
#pragma HLS INLINE
    ap_int<ENERGY_BITS> right = energy_bits(b);
    if (subtract) right = ~right;
    return energy_from_bits(prefix_add<ENERGY_BITS>(energy_bits(a), right, subtract));
}

static energy_t signed_term(energy_t value, bool negate) {
#pragma HLS INLINE
    return negate ? energy_add(energy_t(0), value, true) : value;
}

template <int N>
static energy_t sum_energy(energy_t values[N]) {
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable=values complete dim=1
    ap_int<ENERGY_BITS> words[N];
#pragma HLS ARRAY_PARTITION variable=words complete dim=1
    for (int i = 0; i < N; ++i) {
#pragma HLS UNROLL
        words[i] = energy_bits(values[i]);
    }
    return energy_from_bits(wallace_sum<N, ENERGY_BITS>(words));
}

static int edge_index(int i, int j) {
#pragma HLS INLINE
    const int a = i < j ? i : j;
    const int b = i < j ? j : i;
    return a * (2 * SK_N - a - 1) / 2 + b - a - 1;
}

static prob_t metropolis_probability(energy_t delta) {
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable=CHEB_COEFFS complete dim=1
    if (delta <= energy_t(0)) return prob_t(1);
    // Above the cutoff, exp(-beta*delta) is at most EPS_OP/2.
    if (delta >= EXP_E_CUT) return prob_t(0);

    poly_t y = poly_t(EXP_Y_SLOPE * poly_t(delta) - poly_t(1));
    poly_t b1 = 0;
    poly_t b2 = 0;
    for (int k = POLY_DEGREE; k >= 1; --k) {
#pragma HLS UNROLL
        poly_t b0 = poly_t(poly_t(2) * y * b1 - b2 + CHEB_COEFFS[k]);
        b2 = b1;
        b1 = b0;
    }
    poly_t p = poly_t(CHEB_COEFFS[0] + y * b1 - b2);
    if (p <= poly_t(0)) return prob_t(0);
    if (p >= poly_t(1)) return prob_t(1);
    return prob_t(p);
}

void local_spin_flip_operation(const coeff_t h[SK_N], const coeff_t J[NUM_J],
    ap_uint<SK_N> spins, ap_uint<SK_N> flip_mask,
    energy_t* delta_energy, prob_t* accept_prob);

void uniform_move_operation(const coeff_t h[SK_N], const coeff_t J[NUM_J],
    ap_uint<SK_N> proposed_spins, energy_t old_energy,
    energy_t* new_energy, energy_t* delta_energy, prob_t* accept_prob);
