#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using namespace std;

int n;
vector<double> h;
vector<double> J;
volatile long double compiler_guard = 0.0L;

int jidx(int i, int j) {
    return i * n - i * (i + 1) / 2 + (j - i - 1);
}

double Jij(int i, int j) {
    return J[jidx(i, j)];
}

double energy(uint64_t x) {
    int8_t s[64];
    double E = 0.0;

    for(int i = 0; i < n; i++) {
        s[i] = ((x >> i) & 1ULL) ? -1 : 1;
    }

    for(int i = 0; i < n; i++) {
        E -= h[i] * (double)s[i];
    }

    for(int i = 0; i < n; i++) {
        for(int j = i + 1; j < n; j++) {
            E -= Jij(i, j) * (double)(s[i] * s[j]);
        }
    }

    return E;
}

void fill_vectors(const char* filename) {
    ifstream in(filename);
    if(!in) {
        cerr << "cannot open input file: " << filename << "\n";
        exit(1);
    }

    h.assign(n, 0.0);
    J.assign(n * (n - 1) / 2, 0.0);

    for(int i = 0; i < n; i++) {
        if(!(in >> h[i])) {
            cerr << "malformed input file while reading h\n";
            exit(1);
        }
    }

    for(int i = 0; i < n * (n - 1) / 2; i++) {
        if(!(in >> J[i])) {
            cerr << "malformed input file while reading J\n";
            exit(1);
        }
    }
}

long double enumerate_once(uint64_t N) {
    long double sum = 0.0L;

    for(uint64_t x = 0; x < N; x++) {
        sum += (long double)energy(x);
    }

    compiler_guard = sum;
    return sum;
}

int main(int argc, char** argv) {
    cout << unitbuf;

    if(argc < 3) {
        cerr << "usage: ./estimation_enumeration_new2.x <n> <filename.txt>\n";
        return 1;
    }

    n = stoi(argv[1]);
    const char* filename = argv[2];

    if(n <= 0 || n > 30) {
        cerr << "this exact enumeration timing code allows 1 <= n <= 30\n";
        return 1;
    }

    fill_vectors(filename);

    uint64_t N = 1ULL << n;

    auto t0 = chrono::steady_clock::now();
    long double sum = enumerate_once(N);
    auto t1 = chrono::steady_clock::now();

    double seconds_total = chrono::duration<double>(t1 - t0).count();
    double seconds_per_operation = seconds_total / (double)N;
    double ns_per_operation = 1.0e9 * seconds_per_operation;

    cout << setprecision(21);
    cout << "n " << n << "\n";
    cout << "input_file " << filename << "\n";
    cout << "states " << N << "\n";
    cout << "sum " << sum << "\n";
    cout << "seconds_total " << seconds_total << "\n";
    cout << "seconds_per_operation " << seconds_per_operation << "\n";
    cout << "ns_per_operation " << ns_per_operation << "\n";
    cout << "completed 1\n";

    return 0;
}
