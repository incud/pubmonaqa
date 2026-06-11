#include <ap_fixed.h>
#include <ap_int.h>

#ifndef SK_N
#error "SK_N is not defined. Compile with -DSK_N=<number_of_spins>."
#endif

#ifndef FRAC
#error "FRAC is not defined. Compile with -DFRAC=<bits> where FRAC = ceil(3.5 * log2(SK_N / EPS_DISCR))."
#endif

// Number of independent SK couplings J_ij with i < j.
// The input J_edges stores only the upper-triangular part of J, packed into a flat array.
static const int NUM_J = SK_N * (SK_N - 1) / 2;

// Number of terms in the full dense SK energy:
//   - SK_N field terms;
//   - NUM_J pair-coupling terms.
static const int NUM_ENERGY_TERMS = SK_N + NUM_J;

// Q1.F signed fixed-point coefficient: one sign/integer bit and F fractional bits.
// The input coefficients h_i and J_ij are assumed to be already divided by alpha.
using coeff_t = ap_fixed<FRAC + 1, 1, AP_RND, AP_SAT>;

// Uniform-move energy difference type. Every energy difference/alpha is in [-1, 1)
// so only one bit of precision is required. 
using delta_t = ap_fixed<FRAC + 1, 1, AP_RND, AP_SAT>;

// Acceptance probability type. This is unsigned because probabilities lie in [0,1].
// The single integer bit is needed to represent the value 1 exactly.
using prob_t = ap_ufixed<32, 1, AP_RND, AP_SAT>;

// Signed polynomial arithmetic type. Although the final probability is nonnegative,
// Chebyshev coefficients and Clenshaw intermediate values can be negative.
using poly_t = ap_fixed<40, 4, AP_RND, AP_SAT>;

// Scale type for EXP_INV_X_CUT = 1/x_cut = beta*alpha/tau.
// This can be much larger than 1, so it must not use poly_t<...,4>.
using scale_t = ap_ufixed<48, 16, AP_RND, AP_SAT>;

// Generated offline by 'make_exp_cheby.py' for fixed SK_N, beta, polynomial degree,
// and operation-error budget. The generated header implements the piecewise-tail
// approximation of g(x)=exp(-beta*alpha*x):
//   g(x)=1 for x<=0,
//   g(x)=p_d(2*x/x_cut-1) for 0<x<x_cut,
//   g(x)=0 for x>=x_cut,
// where x_cut=tau/(beta*alpha) and tau=log(1/eps_tail).
// Must define:
//   static const int POLY_DEGREE;
//   static const delta_t EXP_X_CUT;
//   static const scale_t EXP_INV_X_CUT;
//   static const poly_t CHEB_COEFFS[POLY_DEGREE + 1];
#include "exp_cheby.hpp"

// Convert a Boolean spin bit to an Ising spin in {-1,+1}.
// Convention: false -> -1 and true -> +1.
static delta_t spin_value(bool b) {
#pragma HLS INLINE
    return b ? delta_t(1) : delta_t(-1);
}

// Return the packed upper-triangular index of J_ab, where a=min(i,j), b=max(i,j).
// This lets the circuit read J_ij from the flat J_edges array instead of a full SK_N x SK_N matrix.
static int edge_index(int i, int j) {
#pragma HLS INLINE
    int a = i < j ? i : j;
    int b = i < j ? j : i;
    return a * (2 * SK_N - a - 1) / 2 + (b - a - 1);
}

// Fully unrolled binary tree reduction.
// The input array x is partitioned completely, so all entries can be read in parallel.
// At each recursive level, adjacent pairs are added in parallel, reducing N values to ceil(N/2).
// Repeating this recursively gives logarithmic adder depth instead of a sequential accumulation chain.
template <int N>
static delta_t tree_sum(delta_t x[N]) {
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable=x complete dim=1

    if constexpr (N == 1) {
        return x[0];
    } else {
        static const int M = (N + 1) / 2;
        delta_t y[M];
#pragma HLS ARRAY_PARTITION variable=y complete dim=1

        for (int k = 0; k < M; ++k) {
#pragma HLS UNROLL
            if (2 * k + 1 < N) {
                y[k] = delta_t(x[2 * k] + x[2 * k + 1]);
            } else {
                y[k] = x[2 * k];
            }
        }

        return tree_sum<M>(y);
    }
}

// Return -a*s_i. This is the field contribution to the SK energy
//   E(s)/alpha = -sum_i (h_i/alpha)s_i - sum_{i<j}(J_ij/alpha)s_i s_j.
static delta_t field_delta_term(coeff_t a, bool spin_i) {
#pragma HLS INLINE
    delta_t x = delta_t(a);
    delta_t neg_x = delta_t(0) - x;
    return spin_i ? neg_x : x;
}

// Return -J_ij*s_i*s_j. If the two spins are equal, s_i*s_j=+1 and the term is -J_ij.
// If they differ, s_i*s_j=-1 and the term is +J_ij.
static delta_t pair_delta_term(coeff_t Jij, bool spin_i, bool spin_j) {
#pragma HLS INLINE
    delta_t x = delta_t(Jij);
    delta_t neg_x = delta_t(0) - x;
    return (spin_i == spin_j) ? neg_x : x;
}

// Compute the full dense normalized SK energy E(spins)/alpha for a proposed configuration.
// Unlike the local move, this is an O(N^2) dense calculation: all h_i terms and all J_ij
// pair terms are formed in parallel and then reduced by a log-depth adder tree.
static delta_t dense_energy_over_alpha(
    const coeff_t h[SK_N],
    const coeff_t J_edges[NUM_J],
    ap_uint<SK_N> spins
) {
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable=h complete dim=1
#pragma HLS ARRAY_PARTITION variable=J_edges complete dim=1

    delta_t terms[NUM_ENERGY_TERMS];
#pragma HLS ARRAY_PARTITION variable=terms complete dim=1

    // Field terms: -h_i s_i.
    for (int i = 0; i < SK_N; ++i) {
#pragma HLS UNROLL
        terms[i] = field_delta_term(h[i], spins[i]);
    }

    // Pair terms: -J_ij s_i s_j.
    for (int i = 0; i < SK_N; ++i) {
#pragma HLS UNROLL
        for (int j = i + 1; j < SK_N; ++j) {
#pragma HLS UNROLL
            int e = edge_index(i, j);
            terms[SK_N + e] = pair_delta_term(J_edges[e], spins[i], spins[j]);
        }
    }

    return tree_sum<NUM_ENERGY_TERMS>(terms);
}

// Map delta_over_alpha = (E_new - E_old)/alpha to x=max(delta_over_alpha,0).
// The raw uniform-move difference can be in (-2,2), so the input is delta_t.
// The output is saturated into the Q1.F delta_t range used by chebyshev_g.
// If the positive difference is larger than the representable range, saturation
// is harmless for the probability path because x will be far above x_cut and g(x)=0.
static delta_t positive_part(delta_t delta_over_alpha) {
#pragma HLS INLINE
    if (delta_over_alpha <= delta_t(0)) {
        return delta_t(0);
    }

    return delta_t(delta_over_alpha);
}

// Evaluate the generated piecewise Chebyshev approximation to g(x)=exp(-beta*alpha*x).
// The polynomial is not a global approximation on [0,1]. It is evaluated only on
// the short interval [0,x_cut], and the output is forced to zero for x>=x_cut.
static prob_t chebyshev_g(delta_t x) {
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable=CHEB_COEFFS complete dim=1

    if (x <= delta_t(0)) {
        return prob_t(1);
    }

    if (x >= EXP_X_CUT) {
        return prob_t(0);
    }

    // Map x in [0,x_cut] to z=x/x_cut in [0,1], then y=2z-1 in [-1,1].
    scale_t z = scale_t(x) * EXP_INV_X_CUT;
    poly_t y = poly_t(2) * poly_t(z) - poly_t(1);

    // Clenshaw recurrence for p(y)=sum_k c_k T_k(y).
    poly_t b1 = poly_t(0);
    poly_t b2 = poly_t(0);

    for (int k = POLY_DEGREE; k >= 1; --k) {
#pragma HLS UNROLL
        poly_t b0 = poly_t(poly_t(2) * y * b1 - b2 + CHEB_COEFFS[k]);
        b2 = b1;
        b1 = b0;
    }

    poly_t p = poly_t(CHEB_COEFFS[0] + y * b1 - b2);

    if (p <= poly_t(0)) {
        return prob_t(0);
    }

    if (p >= poly_t(1)) {
        return prob_t(1);
    }

    return prob_t(p);
}

/**
 * Top-level uniform-move operation kernel.
 *
 * This is the HLS entry point for a dense non-local proposal. It does not use the
 * local single-spin delta-energy identity. Instead it:
 *   1. computes new_energy = E(new_spins)/alpha by a full dense O(N^2) energy calculation;
 *   2. computes delta_over_alpha = new_energy - old_energy;
 *   3. clips it to x=max(delta_over_alpha,0);
 *   4. evaluates the pre-generated piecewise Chebyshev approximation to exp(-beta*alpha*x).
 *
 * Inputs:
 *   h[SK_N]
 *       Fields already normalized by alpha, stored as Q1.F fixed-point values.
 *   J_edges[NUM_J]
 *       Upper-triangular couplings already normalized by alpha, packed with edge_index(i,j).
 *   new_spins
 *       Proposed new spin configuration. Bit 1 means +1, bit 0 means -1.
 *   old_energy
 *       Current normalized energy E(old_spins)/alpha, supplied by the caller.
 *
 * Outputs:
 *   new_energy
 *       Newly computed normalized energy E(new_spins)/alpha.
 *   delta_x
 *       Nonnegative normalized value x=max((E(new)-E(old))/alpha,0).
 *   accept_prob
 *       Piecewise approximation to exp(-beta*alpha*x), clipped to [0,1].
 */
void uniform_move_operation(
    const coeff_t h[SK_N],
    const coeff_t J_edges[NUM_J],
    ap_uint<SK_N> new_spins,
    delta_t old_energy,
    delta_t* new_energy,
    delta_t* delta_x,
    prob_t* accept_prob
) {
#pragma HLS ARRAY_PARTITION variable=h complete dim=1
#pragma HLS ARRAY_PARTITION variable=J_edges complete dim=1

#pragma HLS INTERFACE ap_none port=new_spins
#pragma HLS INTERFACE ap_none port=old_energy
#pragma HLS INTERFACE ap_none port=new_energy
#pragma HLS INTERFACE ap_none port=delta_x
#pragma HLS INTERFACE ap_none port=accept_prob
#pragma HLS INTERFACE ap_ctrl_none port=return

    delta_t e_new = dense_energy_over_alpha(h, J_edges, new_spins);
    delta_t delta_over_alpha = delta_t(e_new) - delta_t(old_energy);
    delta_t x = positive_part(delta_over_alpha);
    prob_t p = chebyshev_g(x);

    *new_energy = e_new;
    *delta_x = x;
    *accept_prob = p;
}
