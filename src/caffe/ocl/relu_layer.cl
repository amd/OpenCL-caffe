template <class T>
__kernel void ReLUForward(const int count, __global T* in, __global T* out, T negative_slope){
	int index = get_global_id(0);
	if(index < count)
		out[index] = in[index] > 0? in[index]:in[index]*negative_slope;
}

template __attribute__ ((mangled_name(ReLUForwardFloat))) __kernel void ReLUForward(const int count, __global float* in, __global float* out, float negative_slope);
template __attribute__ ((mangled_name(ReLUForwardDouble))) __kernel void ReLUForward(const int count, __global double* in, __global double* out, double negative_slope);

template <class T>
__kernel void ReLUBackward(const int count, __global T* in_diff, __global T* in_data,__global T* out_diff,T negative_slope){
	int index = get_global_id(0);
        if(index < count)
		out_diff[index] = in_diff[index] * (in_data[index] > 0)+(in_data[index] <= 0) * negative_slope;
}

template __attribute__ ((mangled_name(ReLUBackwardFloat))) __kernel void ReLUBackward(const int count, __global float* in_diff, __global float* in_data, __global float* out_diff, float negative_slope);
template __attribute__ ((mangled_name(ReLUBackwardDouble))) __kernel void ReLUBackward(const int count, __global double* in_diff, __global double* in_data, __global double* out_diff, double negative_slope);
