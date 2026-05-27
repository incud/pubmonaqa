import math

LUT_BITS = 10
LUT_SIZE = 1 << LUT_BITS
X_MAX = 32.0

print("#ifndef EXP_NEG_LUT_HPP")
print("#define EXP_NEG_LUT_HPP")
print()
print(f"static const int EXP_NEG_LUT_SIZE = {LUT_SIZE};")
print(f"static const coeff_t EXP_NEG_XMAX = coeff_t(\"{X_MAX:.18e}\");")
print()
print("static const prob_t EXP_NEG_LUT[EXP_NEG_LUT_SIZE + 1] = {")
for k in range(LUT_SIZE + 1):
    x = X_MAX * k / LUT_SIZE
    y = math.exp(-x)
    comma = "," if k < LUT_SIZE else ""
    print(f"    prob_t(\"{y:.18e}\"){comma}")
print("};")
print()
print("#endif")
