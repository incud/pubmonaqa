#pragma once
// Generated: FRAC rounds physical Hamiltonian coefficients.
#ifndef SK_N
#error "Compile with -DSK_N=<configured size>"
#endif
#if SK_N == 8
#define FRAC 18
#define INT_BITS 6
#define MAX_DELTA_ENERGY 26
#define EXP_CUT_VALUE 9.9034881591796875
#elif SK_N == 10
#define FRAC 19
#define INT_BITS 7
#define MAX_DELTA_ENERGY 36
#define EXP_CUT_VALUE 9.9034881591796875
#elif SK_N == 16
#define FRAC 20
#define INT_BITS 8
#define MAX_DELTA_ENERGY 74
#define EXP_CUT_VALUE 9.9034881591796875
#elif SK_N == 32
#define FRAC 22
#define INT_BITS 9
#define MAX_DELTA_ENERGY 206
#define EXP_CUT_VALUE 9.9034876823425293
#elif SK_N == 64
#define FRAC 24
#define INT_BITS 11
#define MAX_DELTA_ENERGY 578
#define EXP_CUT_VALUE 9.9034875631332397
#else
#error "SK_N is not in this run; regenerate the headers"
#endif
