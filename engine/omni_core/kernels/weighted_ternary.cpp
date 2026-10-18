#include <cstdint>
#include <limits>

#if defined(_WIN32)
#define OMNI_EXPORT extern "C" __declspec(dllexport)
#else
#define OMNI_EXPORT extern "C" __attribute__((visibility("default")))
#endif

// Four native ternary signs per byte, with independent strength in a byte's
// low nibble. Decode only the current value into a register, never a weight
// matrix. The high nibble is learning resistance, not a forward multiplier.
static inline int weight_at(const uint8_t* signs, const uint8_t* states,
                            int64_t row, int64_t column, int64_t width) {
    const int code = (signs[row * ((width + 3) / 4) + column / 4] >> (2 * (column % 4))) & 3;
    if (code == 3) return 32; // Reserved code, distinct from every valid value.
    const int strength = states ? (states[row * width + column] & 15) + 1 : 1;
    return (code - 1) * strength;
}

OMNI_EXPORT int omni_packed_kernel_abi() { return 1; }

OMNI_EXPORT int omni_weighted_ternary_forward_i8(
    const int8_t* input, const uint8_t* signs, const uint8_t* states,
    int64_t samples, int64_t width, int64_t rows, int64_t* output) {
    if (samples < 0 || width < 0 || rows < 0 || width > std::numeric_limits<int64_t>::max() / 2048) return 2;
    for (int64_t sample = 0; sample < samples; ++sample) {
        for (int64_t row = 0; row < rows; ++row) {
            int64_t sum = 0;
            for (int64_t column = 0; column < width; ++column) {
                const int weight = weight_at(signs, states, row, column, width);
                if (weight == 32) return 1;
                sum += static_cast<int64_t>(input[sample * width + column]) * weight;
            }
            output[sample * rows + row] = sum;
        }
    }
    return 0;
}

OMNI_EXPORT int omni_weighted_ternary_input_gradient_f32(
    const float* gradient, const uint8_t* signs, const uint8_t* states,
    int64_t samples, int64_t width, int64_t rows, float* output) {
    if (samples < 0 || width < 0 || rows < 0) return 2;
    for (int64_t sample = 0; sample < samples; ++sample) {
        for (int64_t column = 0; column < width; ++column) {
            double sum = 0;
            for (int64_t row = 0; row < rows; ++row) {
                const int weight = weight_at(signs, states, row, column, width);
                if (weight == 32) return 1;
                sum += static_cast<double>(gradient[sample * rows + row]) * weight;
            }
            output[sample * width + column] = static_cast<float>(sum);
        }
    }
    return 0;
}

// Activity stays floating point: recurrent rates are not quantized to int8.
// Only one sign/gain is decoded in a register. No dense weight tile exists.
template <typename Value>
static int weighted_ternary_forward_float(
    const Value* input, const uint8_t* signs, const uint8_t* states,
    int64_t samples, int64_t width, int64_t rows, Value* output) {
    if (samples < 0 || width < 0 || rows < 0 || width > std::numeric_limits<int64_t>::max() - 3) return 2;
    const int64_t stride = (width + 3) / 4;
    if ((stride && rows > std::numeric_limits<int64_t>::max() / stride)
        || (width && rows > std::numeric_limits<int64_t>::max() / width)
        || (width && samples > std::numeric_limits<int64_t>::max() / width)
        || (rows && samples > std::numeric_limits<int64_t>::max() / rows)) return 2;
    // Include unused padding lanes in reserved-code validation, matching the
    // old native-router tile reader without expanding the packed bytes.
    for (int64_t offset = 0; offset < rows * stride; ++offset) {
        const uint8_t byte = signs[offset];
        for (int shift = 0; shift < 8; shift += 2)
            if (((byte >> shift) & 3) == 3) return 1;
    }
    for (int64_t sample = 0; sample < samples; ++sample) {
        for (int64_t row = 0; row < rows; ++row) {
            double sum = 0.0;
            for (int64_t column = 0; column < width; ++column) {
                const int weight = weight_at(signs, states, row, column, width);
                sum += static_cast<double>(input[sample * width + column]) * weight;
            }
            output[sample * rows + row] = static_cast<Value>(sum);
        }
    }
    return 0;
}

OMNI_EXPORT int omni_weighted_ternary_forward_f32(
    const float* input, const uint8_t* signs, const uint8_t* states,
    int64_t samples, int64_t width, int64_t rows, float* output) {
    return weighted_ternary_forward_float(input, signs, states, samples, width, rows, output);
}

OMNI_EXPORT int omni_weighted_ternary_forward_f64(
    const double* input, const uint8_t* signs, const uint8_t* states,
    int64_t samples, int64_t width, int64_t rows, double* output) {
    return weighted_ternary_forward_float(input, signs, states, samples, width, rows, output);
}
