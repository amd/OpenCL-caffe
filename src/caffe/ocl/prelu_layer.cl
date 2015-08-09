template <class T>
__kernel void PReLUForward(const int count, const int channels, const int dim, __global T* in, __global T* out, __global T* slope_data, const int div_factor) {
  int index = get_global_id(0);
  if(index < count){
    int c = (index / dim) % channels / div_factor;
    out[index] = in[index] > 0 ? in[index] : in[index] * slope_data[c];
  }
}
template __attribute__ ((mangled_name(PReLUForward_float))) __kernel void PReLUForward(const int count, const int channels, const int dim, __global float* in, __global float* out, __global float* slope_data, const int div_factor);
template __attribute__ ((mangled_name(PReLUForward_double))) __kernel void PReLUForward(const int count, const int channels, const int dim, __global double* in, __global double* out, __global double* slope_data, const int div_factor);

template <class T>
__kernel void PReLUBackward(const int count, const int channels, const int dim, __global T* in_diff, __global T* in_data, __global T* out_diff, __global T* slope_data, const int div_factor) {
  int index = get_global_id(0);
  if(index < count){
    int c = (index / dim) % channels / div_factor;
    out_diff[index] = in_diff[index] * ((in_data[index] > 0)
        + (in_data[index] <= 0) * slope_data[c]);
  }
}
template __attribute__ ((mangled_name(PReLUBackward_float))) __kernel void PReLUBackward(const int count, const int channels, const int dim, __global float* in_diff, __global float* in_data, __global float* out_diff, __global float* slope_data, const int div_factor);
template __attribute__ ((mangled_name(PReLUBackward_double))) __kernel void PReLUBackward(const int count, const int channels, const int dim, __global double* in_diff, __global double* in_data, __global double* out_diff, __global double* slope_data, const int div_factor);

template <class T>
__kernel void PReLUParamBackward(const int count, __global T* in_diff, __global T* in_data, __global T* out_diff) {
  int index = get_global_id(0);
  if(index < count){
    out_diff[index] = in_diff[index] * in_data[index] * (in_data[index] <= 0);
  }
}
template __attribute__ ((mangled_name(PReLUParamBackward_float))) __kernel void PReLUParamBackward(const int count, __global float* in_diff, __global float* in_data, __global float* out_diff);
template __attribute__ ((mangled_name(PReLUParamBackward_double))) __kernel void PReLUParamBackward(const int count, __global double* in_diff, __global double* in_data, __global double* out_diff);
