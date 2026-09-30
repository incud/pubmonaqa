#include <bits/stdc++.h>

using namespace std;

int n;
vector<double> h;
vector<double> J;

const long double minimum_overlap = exp(-1.0L);

int jidx(int i, int j) {
    return i * n - i * (i + 1) / 2 + (j - i - 1);
}

double Jij(int i, int j) {
    if (i > j) swap(i, j);
    return J[jidx(i, j)];
}

void fill_vectors(char* filename) {
    ifstream in(filename);
    if (!in) {
        cerr << "cannot open input file\n";
        exit(1);
    }

    h.assign(n, 0.0);
    J.assign(n * (n - 1) / 2, 0.0);

    for (int i = 0; i < n; i++) in >> h[i];
    for (int i = 0; i < n * (n - 1) / 2; i++) in >> J[i];
}

double v(double beta) {
    constexpr double v1 = 0.36509, kappa = 1.23536, tau = 1.28646;
    if (beta < 1.0) return exp(log(v1) * beta + kappa * beta * (1.0 - beta));
    return v1 * pow(1.0 + (beta - 1.0) / tau, -3.5);
}

double delta(double beta, double s) {
    return 2.0 / sqrt(s * n * v(beta));
}

int flipped_spin(uint64_t step) {
    // Spin flipped at this step of the Gray-code enumeration.
    int i = 0;
    while ((step & 1) == 0) {
        step >>= 1;
        i++;
    }
    return i;
}

long double overlap_squared(double beta, double next) {
    vector<int> spin(n, 1);
    vector<long double> field(n);
    long double energy = 0.0L;

    // Initial all-+1 configuration.
    for (int i = 0; i < n; i++) {
        field[i] = h[i];
        energy -= h[i];
        for (int j = 0; j < n; j++)
            if (i != j) field[i] += Jij(i, j);
    }

    for (int i = 0; i < n; i++)
        for (int j = i + 1; j < n; j++)
            energy -= Jij(i, j);

    double middle = 0.5 * (beta + next);
    uint64_t N = 1ULL << n;

    // Stream the three partition functions without storing the 2^n energies.
    long double emin = energy;
    long double z_beta = 1.0L, z_middle = 1.0L, z_next = 1.0L;

    for (uint64_t step = 1; step < N; step++) {
        int i = flipped_spin(step);
        long double old_spin = spin[i];

        // Energy change caused by flipping spin i.
        energy += 2.0L * old_spin * field[i];

        // Update local fields after the flip.
        for (int j = 0; j < n; j++)
            if (j != i) field[j] -= 2.0L * old_spin * Jij(i, j);

        spin[i] = -spin[i];

        // Keep the partition-function sums numerically stable.
        if (energy < emin) {
            long double d = emin - energy;
            z_beta = z_beta * exp(-(long double)beta * d) + 1.0L;
            z_middle = z_middle * exp(-(long double)middle * d) + 1.0L;
            z_next = z_next * exp(-(long double)next * d) + 1.0L;
            emin = energy;
        } else {
            long double d = energy - emin;
            z_beta += exp(-(long double)beta * d);
            z_middle += exp(-(long double)middle * d);
            z_next += exp(-(long double)next * d);
        }
    }

    return min(1.0L, z_middle * z_middle / (z_beta * z_next));
}

double binary_search(double beta, double target) {
    // Find the smallest passing safety factor with resolution 0.1.
    uint64_t lo = 10, hi = 20;

    while (true) {
        double s = hi / 10.0;
        double next = min(target, beta + delta(beta, s));
        if (beta < 1.0 && next > 1.0) next = 1.0;
        if (overlap_squared(beta, next) >= minimum_overlap) break;
        lo = hi;
        hi *= 2;
    }

    while (hi - lo > 1) {
        uint64_t mid = lo + (hi - lo) / 2;
        double s = mid / 10.0;
        double next = min(target, beta + delta(beta, s));
        if (beta < 1.0 && next > 1.0) next = 1.0;

        if (overlap_squared(beta, next) >= minimum_overlap) hi = mid;
        else lo = mid;
    }

    return hi / 10.0;
}

vector<tuple<double, long double, double>> tight_schedule_safe(double target = 16.0) {
    vector<tuple<double, long double, double>> schedule;

    for (double beta = 0.0; beta < target;) {
        double safety = 1.0;
        double next = min(target, beta + delta(beta, safety));
        if (beta < 1.0 && next > 1.0) next = 1.0;

        long double overlap = overlap_squared(beta, next);

        if (overlap < minimum_overlap) {
            safety = binary_search(beta, target);
            next = min(target, beta + delta(beta, safety));
            if (beta < 1.0 && next > 1.0) next = 1.0;
            overlap = overlap_squared(beta, next);
        }

        schedule.emplace_back(beta, overlap, safety);
        beta = next;
    }

    return schedule;
}

vector<tuple<double, long double, double>> fixed_schedule(double safety, double target = 16.0) {
    vector<tuple<double, long double, double>> schedule;

    for (double beta = 0.0; beta < target;) {
        double next = min(target, beta + delta(beta, safety));
        if (beta < 1.0 && next > 1.0) next = 1.0;
        schedule.emplace_back(beta, overlap_squared(beta, next), safety);
        beta = next;
    }

    return schedule;
}

int main(int argc, char** argv) {
    cout << unitbuf;

    if (argc != 3 && argc != 5) {
        cerr << "usage: ./generation_annealing_schedules <n> <filename.txt> [--fixed-safety <s>]\n";
        return 1;
    }

    n = stoi(argv[1]);

    if (n <= 0 || n > 30) {
        cerr << "this exact streaming code allows 1 <= n <= 30\n";
        return 1;
    }

    fill_vectors(argv[2]);

    bool fixed = argc == 5;
    if (fixed && string(argv[3]) != "--fixed-safety") {
        cerr << "expected --fixed-safety <s>\n";
        return 1;
    }

    double safety = fixed ? stod(argv[4]) : 1.0;
    if (safety <= 0.0) {
        cerr << "safety factor must be positive\n";
        return 1;
    }

    auto schedule = fixed ? fixed_schedule(safety) : tight_schedule_safe();

    cout << setprecision(17);
    cout << "# beta overlap_squared safety_margin\n";

    for (auto [beta, overlap, safety] : schedule)
        cout << beta << " " << overlap << " " << safety << "\n";

    return 0;
}