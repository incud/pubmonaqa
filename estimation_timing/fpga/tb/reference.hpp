#pragma once
#include "fpga.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

static void require(bool ok, const std::string& message) {
    if (!ok) throw std::runtime_error(message);
}

static int reference_index(int i, int j) {
    if (i > j) std::swap(i, j);
    return i * (2 * SK_N - i - 1) / 2 + j - i - 1;
}

static void make_model(coeff_t h[SK_N], coeff_t J[NUM_J], unsigned seed) {
    std::mt19937 random(seed);
    std::normal_distribution<double> gaussian(0, 1);
    std::vector<double> coefficients(NUM_TERMS);
    double norm2 = 0;
    for (double& c : coefficients) {
        c = gaussian(random);
        norm2 += c * c;
    }
    const double model_factor = std::sqrt(double(SK_N) / norm2);
    const double rounding_limit = std::ldexp(1.0, -FRAC - 1);
    double one_norm = 0;
    for (int i = 0; i < NUM_TERMS; ++i) {
        double physical = coefficients[i] * model_factor;
        coeff_t rounded = coeff_t(physical);
        require(std::abs(rounded.to_double() - physical) <= rounding_limit * (1 + 1e-9),
                "Coefficient rounding is inconsistent with the physical-energy grid");
        if (i < SK_N) h[i] = rounded;
        else J[i - SK_N] = rounded;
        one_norm += std::abs(rounded.to_double());
    }
    require(one_norm < std::ldexp(1.0, INT_BITS - 1), "Energy partial sums exceed the signed type");
}

static double reference_energy(const coeff_t h[SK_N], const coeff_t J[NUM_J],
                               ap_uint<SK_N> spins) {
    double energy = 0;
    for (int i = 0; i < SK_N; ++i) {
        int si = spins[i] ? 1 : -1;
        energy -= h[i].to_double() * si;
        for (int j = i + 1; j < SK_N; ++j)
            energy -= J[reference_index(i, j)].to_double() * si * (spins[j] ? 1 : -1);
    }
    return energy;
}

static ap_uint<SK_N> configuration(int id, std::mt19937_64& random) {
    if constexpr (SK_N <= 10) return ap_uint<SK_N>(id);
    if (id == 0) return ap_uint<SK_N>(0);
    if (id == 1) return ~ap_uint<SK_N>(0);
    return ap_uint<SK_N>(random());
}

static void check_probability(double delta, prob_t probability, double& largest_error) {
    const double expected = delta <= 0 ? 1 : std::exp(-BETA * delta);
    const double error = std::abs(probability.to_double() - expected);
    largest_error = std::max(largest_error, error);
    require(error <= EPS_OP, "Acceptance approximation exceeds the per-evaluation tolerance");
    if (delta <= 0) require(probability == prob_t(1), "A downhill/zero move must be accepted exactly");
}

static void check_probability_boundaries(double& largest_error) {
    // Cover the whole supported interval, including its steep small-energy part.
    for (int i = -1; i <= 4096; ++i) {
        energy_t delta = energy_t(double(MAX_DELTA_ENERGY) * i / 4096);
        check_probability(delta.to_double(), metropolis_probability(delta), largest_error);
    }
    for (int i = 0; i <= 4096; ++i) {
        energy_t delta = energy_t(16.0 * i / 4096);
        check_probability(delta.to_double(), metropolis_probability(delta), largest_error);
    }
    for (int units : {-1, 0, 1}) {
        energy_t delta = energy_t(std::ldexp(double(units), -FRAC));
        check_probability(delta.to_double(), metropolis_probability(delta), largest_error);
        delta = energy_t(EXP_E_CUT + energy_t(std::ldexp(double(units), -FRAC)));
        prob_t probability = metropolis_probability(delta);
        check_probability(delta.to_double(), probability, largest_error);
        if (units >= 0) require(probability == prob_t(0), "The cutoff must return zero");
    }
}
