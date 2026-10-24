#pragma once

constexpr int ceil_log2(int x) {
    int bits = 0;
    for (int v = x - 1; v > 0; v >>= 1) ++bits;
    return bits;
}

// Parallel-prefix carry propagation: O(log W) logic levels, including carry_in.
template <int W, class Word>
static Word prefix_add(Word a, Word b, bool carry_in = false) {
#pragma HLS INLINE
    Word original_p = a ^ b;
    Word p = original_p;
    Word g = a & b;
    g = g | (p & Word(carry_in));  // Inject the input carry at bit zero.
    for (int level = 0; level < ceil_log2(W); ++level) {
#pragma HLS UNROLL
        const int shift = 1 << level;
        g = g | (p & (g << shift));
        p = p & (p << shift);
    }
    return Word(original_p ^ ((g << 1) | Word(carry_in)));
}

// Signed two's-complement sum modulo 2^W. Choose W so the final sum fits.
// Word can be an HLS W-bit integer or a native unsigned test word (use its low W bits).
// Each 3:2 compressor is carry-save; only the final pair propagates carries.
template <int N, int W, class Word>
static Word wallace_sum(Word values[N]) {
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable=values complete dim=1
    static_assert(N > 0 && W > 0, "Positive count and width required");
    if constexpr (N == 1) {
        return values[0];
    } else if constexpr (N == 2) {
        return prefix_add<W>(values[0], values[1]);
    } else {
        constexpr int GROUPS = N / 3;
        constexpr int NEXT = 2 * GROUPS + N % 3;
        Word next[NEXT];
#pragma HLS ARRAY_PARTITION variable=next complete dim=1
        for (int i = 0; i < GROUPS; ++i) {
#pragma HLS UNROLL
            Word a = values[3 * i];
            Word b = values[3 * i + 1];
            Word c = values[3 * i + 2];
            next[2 * i] = a ^ b ^ c;
            next[2 * i + 1] = ((a & b) | (a & c) | (b & c)) << 1;
        }
        for (int i = 0; i < N % 3; ++i) {
#pragma HLS UNROLL
            next[2 * GROUPS + i] = values[3 * GROUPS + i];
        }
        return wallace_sum<NEXT, W>(next);
    }
}
