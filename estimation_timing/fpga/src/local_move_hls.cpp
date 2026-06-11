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

// Q1.F signed fixed-point coefficient: one sign/integer bit and F fractional bits.
// The input coefficients h_i and J_ij are assumed to be already divided by alpha,
// so they (very abundantly) fit in the interval [-1, 1).
// * AP_RND means round to nearest when an operation produces more fractional bits than the target type.
// * AP_SAT means saturate on overflow instead of wrapping around; this is safer for arithmetic circuits.
using coeff_t = ap_fixed<FRAC + 1, 1, AP_RND, AP_SAT>;

// Normalized local arithmetic type. The normalization by alpha is assumed strong enough
// to keep the relevant local partial sums and the final Delta E / alpha in [-1,1),
// so we intentionally use the same tight Q1.F format as the input coefficients.
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
    // #pragma HLS INLINE: inline the function at the call site instead of synthesizing it 
    // as a separate hardware block. This is useful for small helpers to minimize overhead.
#pragma HLS INLINE
    return b ? delta_t(1) : delta_t(-1);
}

// Return the packed upper-triangular index of J_ab, where a=min(i,j), b=max(i,j).
// This lets the circuit read J_ij from the flat J_edges array instead of a full SK_N x SK_N matrix.
// It is used inside local_delta_over_alpha when the selected flipped spin i is paired with each j.
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

    // This pragma means: split the array x into independent scalar registers/wires along dimension 1.
    // If you avoid it, the HLS compiler can decide to implement this as a memory with only a small 
    // number of ports, usually one or two. Then a fully unrolled loop cannot actually read all elements
    // at once, in parallel. 
#pragma HLS ARRAY_PARTITION variable=x complete dim=1

    if constexpr (N == 1) {
        // Base case: the reduction has only one remaining value.
        return x[0];
    } else {
        // Number of values left after one binary-reduction layer.
        static const int M = (N + 1) / 2;
        delta_t y[M];

        // Completely partition y so the next recursive reduction layer can also be parallel.
#pragma HLS ARRAY_PARTITION variable=y complete dim=1

        for (int k = 0; k < M; ++k) {
            // Fully unroll the pairwise additions in this reduction layer, i.e. instantiate 
            // hardware for many (all) loop iterations instead of reusing the same hardware sequentially.
#pragma HLS UNROLL
            // If two inputs are available, add them; if N is odd, forward the last unpaired input.
            if (2 * k + 1 < N) {
                y[k] = delta_t(x[2 * k] + x[2 * k + 1]);
            } else {
                y[k] = x[2 * k];
            }
        }

        // Recursively implement the remaining reduction layers.
        return tree_sum<M>(y);
    }
}

// Extract the spin s_i selected by the one-hot flip mask.
// Since flip_mask is assumed to contain exactly one bit set, this is a bit selection,
// not an arithmetic reduction. The expression (spins & flip_mask) is nonzero exactly
// when the selected spin bit is 1, corresponding to s_i=+1; otherwise s_i=-1.
static delta_t selected_spin(ap_uint<SK_N> spins, ap_uint<SK_N> flip_mask) {
#pragma HLS INLINE
    bool selected_bit = ((spins & flip_mask) != 0);
    return spin_value(selected_bit);
}

// Select the field h_i associated with the one-hot flip mask.
// This is a one-hot multiplexer, not a numerical sum.
static delta_t selected_field(const coeff_t h[SK_N], ap_uint<SK_N> flip_mask) {
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable=h complete dim=1

    delta_t out = delta_t(0);

    for (int i = 0; i < SK_N; ++i) {
#pragma HLS UNROLL
        if (flip_mask[i]) {
            out = delta_t(h[i]);
        }
    }

    return out;
}

// Select the coupling J_ij for fixed j and the one-hot-selected flipped spin i.
// This is also a one-hot multiplexer over the packed upper-triangular J_edges array.
static delta_t selected_coupling_for_j(
    const coeff_t J_edges[NUM_J],
    ap_uint<SK_N> flip_mask,
    int j
) {
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable=J_edges complete dim=1

    delta_t out = delta_t(0);

    for (int i = 0; i < SK_N; ++i) {
#pragma HLS UNROLL
        if (i != j && flip_mask[i]) {
            out = delta_t(J_edges[edge_index(i, j)]);
        }
    }

    return out;
}

// Compute delta_over_alpha = Delta E / alpha for the selected local spin flip.
// The physical coefficients have already been normalized before entering this circuit:
//   h_input = h_physical / alpha,   J_input = J_physical / alpha.
// Therefore this circuit directly computes Delta E / alpha and never divides by alpha.
//
// For a flip of spin i, the local energy difference is
//   Delta E_i / alpha = 2 s_i (h_i/alpha + sum_{j != i} (J_ij/alpha) s_j).
// The one-hot flip_mask selects i without needing an integer index input.
static delta_t local_delta_over_alpha(
    const coeff_t h[SK_N],
    const coeff_t J_edges[NUM_J],
    ap_uint<SK_N> spins,
    ap_uint<SK_N> flip_mask
) {
    // Inline this helper into the top function so HLS can optimize the full datapath.
#pragma HLS INLINE

    // Complete partition means every h_i can be accessed in parallel.
#pragma HLS ARRAY_PARTITION variable=h complete dim=1

    // Complete partition means every packed J_ij can be accessed in parallel.
#pragma HLS ARRAY_PARTITION variable=J_edges complete dim=1

    // terms contains h_i plus all J_ij s_j terms for the selected flipped spin i.
    delta_t terms[SK_N + 1];
#pragma HLS ARRAY_PARTITION variable=terms complete dim=1

    // Select h_i for the flipped spin i. This is a one-hot mux, not an adder tree.
    terms[0] = selected_field(h, flip_mask);

    // For each neighbor j, select the coupling J_ij associated with the flipped spin i,
    // multiply it by s_j, and store the contribution in terms[j+1].
    for (int j = 0; j < SK_N; ++j) {
#pragma HLS UNROLL
        // Select J_ij for the flipped spin i and the current neighbor j.
        // Since flip_mask is one-hot, this is a mux over J_edges, not a sum.
        delta_t selected_Jij = selected_coupling_for_j(J_edges, flip_mask, j);

        // Add the signed contribution J_ij s_j to the local field.
        terms[j + 1] = delta_t(selected_Jij * spin_value(spins[j]));
    }

    // Compute h_i + sum_{j != i} J_ij s_j using a log-depth tree.
    delta_t field_over_alpha = tree_sum<SK_N + 1>(terms);
    delta_t neg_field_over_alpha = delta_t(0) - field_over_alpha;

    // Final normalized local energy difference: 2 s_i field_i / alpha.
    bool selected_bit = ((spins & flip_mask) != 0);
    delta_t signed_field = selected_bit ? field_over_alpha : neg_field_over_alpha;
    return delta_t(signed_field + signed_field); // 2 * spin * field_over_alpha
}

// Map delta_over_alpha = Delta E / alpha to x = max(delta_over_alpha,0) in [0,1).
// Downhill moves have Delta E < 0, hence x=0 and the Metropolis probability is exactly 1.
// Uphill moves use x=Delta E/alpha and are passed to the Chebyshev approximation.
static delta_t positive_part(delta_t delta_over_alpha) {
#pragma HLS INLINE
    return delta_over_alpha < delta_t(0) ? delta_t(0) : delta_over_alpha;
}

// Evaluate the generated piecewise Chebyshev approximation to g(x)=exp(-beta*alpha*x).
// The polynomial is not a global approximation on [0,1]. It is evaluated only on
// the short interval [0,x_cut], and the output is forced to zero for x>=x_cut.
// This is the hardware version of:
//   u = beta*alpha*x
//   if x <= 0: return 1
//   if u >= tau: return 0
//   return chebval(2*u/tau - 1, coeffs(tau,d)).
static prob_t chebyshev_g(delta_t x) {
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable=CHEB_COEFFS complete dim=1

    // x=0 corresponds to downhill moves after clipping, or zero delta, so g(0)=1.
    if (x <= delta_t(0)) {
        return prob_t(1);
    }

    // For x >= x_cut=tau/(beta*alpha), the exact value is at most eps_tail,
    // so the piecewise approximation returns zero.
    if (x >= EXP_X_CUT) {
        return prob_t(0);
    }

    // Map x in [0,x_cut] to z=x/x_cut in [0,1].
    // EXP_INV_X_CUT can be larger than 1, so this multiplication uses scale_t.
    scale_t z = scale_t(x) * EXP_INV_X_CUT;

    // Map z in [0,1] to the Chebyshev variable y in [-1,1].
    poly_t y = poly_t(2) * poly_t(z) - poly_t(1);

    // Clenshaw recurrence for p(y)=sum_k c_k T_k(y).
    // Here the coefficients approximate exp(-u) on u in [0,tau].
    poly_t b1 = poly_t(0);
    poly_t b2 = poly_t(0);

    for (int k = POLY_DEGREE; k >= 1; --k) {
        // Fully unroll because POLY_DEGREE is a compile-time constant from exp_cheby.hpp.
#pragma HLS UNROLL
        poly_t b0 = poly_t(poly_t(2) * y * b1 - b2 + CHEB_COEFFS[k]);
        b2 = b1;
        b1 = b0;
    }

    // Final Clenshaw combination for the Chebyshev series.
    poly_t p = poly_t(CHEB_COEFFS[0] + y * b1 - b2);

    // Clip the result to [0,1] to enforce a valid probability even after approximation/rounding.
    if (p <= poly_t(0)) {
        return prob_t(0);
    }

    if (p >= poly_t(1)) {
        return prob_t(1);
    }

    return prob_t(p);
}

/**
 * Top-level local spin-flip operation kernel.
 *
 * This is the HLS entry point. It implements only the local single-spin-flip arithmetic:
 *   1. compute delta_over_alpha = Delta E / alpha for the spin selected by flip_mask;
 *   2. clip it to x=max(delta_over_alpha,0);
 *   3. evaluate the pre-generated piecewise Chebyshev approximation to exp(-beta*alpha*x).
 *
 * Inputs:
 *   h[SK_N]
 *       Fields already normalized by alpha, stored as Q1.F fixed-point values.
 *   J_edges[NUM_J]
 *       Upper-triangular couplings already normalized by alpha, packed with edge_index(i,j).
 *   spins
 *       Current spin configuration. Bit 1 means +1, bit 0 means -1.
 *   flip_mask
 *       One-hot mask selecting the spin to flip. The circuit assumes exactly one bit is set.
 *
 * Outputs:
 *   delta_x
 *       Nonnegative normalized value x=max(Delta E/alpha,0). This is the argument of g.
 *   accept_prob
 *       Piecewise approximation to exp(-beta*alpha*x), clipped to [0,1].
 *
 * Convention:
 *   The factor beta*alpha, the cutoff x_cut, and the Chebyshev coefficients are already
 *   absorbed in exp_cheby.hpp. The HLS circuit therefore does not receive beta as an input.
 */
void local_spin_flip_operation(
    const coeff_t h[SK_N],
    const coeff_t J_edges[NUM_J],
    ap_uint<SK_N> spins,
    ap_uint<SK_N> flip_mask,
    delta_t* delta_x,
    prob_t* accept_prob
) {
    // Complete partition exposes all coefficients to the fully parallel selection/reduction logic.
#pragma HLS ARRAY_PARTITION variable=h complete dim=1
#pragma HLS ARRAY_PARTITION variable=J_edges complete dim=1

    // Use simple scalar ports for the bit masks and output pointers in this standalone kernel.
#pragma HLS INTERFACE ap_none port=spins
#pragma HLS INTERFACE ap_none port=flip_mask
#pragma HLS INTERFACE ap_none port=delta_x
#pragma HLS INTERFACE ap_none port=accept_prob

    // Combinational/no-handshake top-level interface. Useful for inspecting the raw datapath latency.
#pragma HLS INTERFACE ap_ctrl_none port=return

    // Signed normalized local energy difference Delta E / alpha.
    delta_t delta_over_alpha = local_delta_over_alpha(h, J_edges, spins, flip_mask);

    // Metropolis clips downhill moves to x=0, making the acceptance probability exactly 1.
    delta_t x = positive_part(delta_over_alpha);

    // Evaluate the offline-generated piecewise polynomial approximation to exp(-beta*alpha*x).
    prob_t p = chebyshev_g(x);

    *delta_x = x;
    *accept_prob = p;
}

