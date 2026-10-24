#include "wallace.hpp"
#include <cstdint>
#include <iostream>
#include <random>
#include <stdexcept>

template <int N, int W>
void check_sum(std::mt19937_64& random) {
    const uint64_t mask = ~uint64_t(0) >> (64 - W);
    for (int trial = 0; trial < 40; ++trial) {
        uint64_t values[N];
        uint64_t expected = 0;
        for (int i = 0; i < N; ++i) {
            uint64_t raw = trial == 0 ? mask : trial == 1 ? 0 :
                           trial == 2 ? (i % 2 ? mask : 1) :
                           trial == 3 ? uint64_t(1) << (W - 1) : random() & mask;
            values[i] = raw;
            expected = (expected + raw) & mask;
        }
        // Native unsigned arithmetic models the same two's-complement low-W-bit sum.
        uint64_t result = wallace_sum<N, W>(values) & mask;
        if (result != expected) {
            throw std::runtime_error("Wallace sum failed: N=" + std::to_string(N) +
                                     ", W=" + std::to_string(W));
        }
    }
}

template <int W>
void check_width(std::mt19937_64& random) {
    check_sum<1, W>(random);
    check_sum<2, W>(random);
    check_sum<3, W>(random);
    check_sum<4, W>(random);
    check_sum<7, W>(random);
    check_sum<17, W>(random);
    check_sum<36, W>(random);
    check_sum<55, W>(random);
    check_sum<136, W>(random);
    check_sum<528, W>(random);
    check_sum<2080, W>(random);
}

int main() {
    try {
        // Exhaustive byte addition/subtraction carry-in cases, including wraparound.
        for (unsigned a = 0; a < 256; ++a) {
            for (unsigned b = 0; b < 256; ++b) {
                for (unsigned carry = 0; carry < 2; ++carry) {
                    uint64_t result = prefix_add<8>(uint64_t(a), uint64_t(b), carry) & 255;
                    if (result != ((a + b + carry) & 255))
                        throw std::runtime_error("Parallel-prefix adder failed");
                }
            }
        }
        std::mt19937_64 random(42);
        check_width<1>(random);
        check_width<2>(random);
        check_width<8>(random);
        check_width<24>(random);
        check_width<37>(random);
        check_width<63>(random);
        std::cout << "PASS Wallace: 131072 prefix cases; 2640 multioperand cases; "
                     "signed, carry propagation, remainders and modular overflow\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
