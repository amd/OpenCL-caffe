#define kBNLL_THRESHOLD  50.0

template <class T>
__kernel void BNLLForward(const int n, __global const T* in, __global T* out) {
  int index = get_global_id(0);
  if (index < n) {
    out[index] = in[index] > 0 ? in[index] + log(1. + exp(-in[index])) : log(1. + exp(in[index]));
  }
}
template __attribute__((mangled_name(BNLLForward_float))) __kernel void BNLLForward(const int n, __global const float* in, __global float* out);
template __attribute__((mangled_name(BNLLForward_double))) __kernel void BNLLForward(const int n, __global const double* in, __global double* out);

template <class T>
__kernel void BNLLBackward(const int n, __global const T* in_diff,
    __global const T* in_data, __global T* out_diff) {
    int index = get_global_id(0);
    if (index < n) {
      T expval = exp(min(in_data[index], T(kBNLL_THRESHOLD)));
      out_diff[index] = in_diff[index] * expval / (expval + 1.);
  }
}

template __attribute__((mangled_name(BNLLBackward_float))) __kernel void BNLLBackward(const int n, __global const float* in_diff,
    __global const float* in_data, __global float* out_diff);
template __attribute__((mangled_name(BNLLBackward_double))) __kernel void BNLLBackward(const int n, __global const double* in_diff,
    __global const double* in_data, __global double* out_diff);
