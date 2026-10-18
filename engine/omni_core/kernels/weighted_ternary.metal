#include <metal_stdlib>
using namespace metal;

inline int packed_weight(device const uchar* signs, device const uchar* states,
                         uint row, uint column, uint width, uint has_state) {
    const uint code = (signs[row * ((width + 3) / 4) + column / 4] >> (2 * (column % 4))) & 3;
    const int strength = has_state ? int(states[row * width + column] & 15) + 1 : 1;
    return (int(code) - 1) * strength;
}

kernel void packed_forward_i8(
    device uint2* output, device const char* activity,
    device const uchar* signs, device const uchar* states,
    device const uint* shape,
    uint lane [[thread_index_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]],
    uint3 group_size [[threads_per_threadgroup]]) {
    const uint width = shape[0], rows = shape[1];
    const uint index = group.x;
    const uint sample = index / rows, row = index % rows;
    threadgroup uint2 partial[128];
    uint low = 0, high = 0;
    for (uint column = lane; column < width; column += group_size.x) {
        const int product = int(activity[sample * width + column]) *
            packed_weight(signs, states, row, column, width, shape[2]);
        const uint previous = low;
        low += uint(product);
        high += product < 0 ? 0xffffffffu : 0u;
        high += low < previous ? 1u : 0u;
    }
    partial[lane] = uint2(low, high);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = group_size.x / 2; stride; stride /= 2) {
        if (lane < stride) {
            const uint2 left = partial[lane], right = partial[lane + stride];
            const uint next_low = left.x + right.x;
            partial[lane] = uint2(next_low, left.y + right.y + uint(next_low < left.x));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    // Exact two's-complement int64 without requiring native GPU int64 math.
    if (lane == 0) output[index] = partial[0];
}

kernel void packed_input_gradient_f32(
    device float* output, device const float* gradient,
    device const uchar* signs, device const uchar* states,
    device const uint* shape, uint index [[thread_position_in_grid]]) {
    const uint width = shape[0], rows = shape[1];
    const uint sample = index / width, column = index % width;
    float sum = 0;
    for (uint row = 0; row < rows; ++row) {
        sum += gradient[sample * rows + row] *
            float(packed_weight(signs, states, row, column, width, shape[2]));
    }
    output[index] = sum;
}

kernel void packed_forward_f32(
    device float* output, device const float* activity,
    device const uchar* signs, device const uchar* states,
    device const uint* shape, uint index [[thread_position_in_grid]]) {
    const uint width = shape[0], rows = shape[1];
    const uint sample = index / rows, row = index % rows;
    float sum = 0;
    // Fixed ascending-column accumulation; activity remains an arbitrary
    // floating rate, while each packed sign and integer gain stays a register.
    for (uint column = 0; column < width; ++column) {
        sum += activity[sample * width + column] *
            float(packed_weight(signs, states, row, column, width, shape[2]));
    }
    output[index] = sum;
}
