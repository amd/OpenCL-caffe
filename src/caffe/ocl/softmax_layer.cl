template <class T>
__kernel void softmax(__global T* prob_data, __global T* loss, __global T* label, int num, int dim, __local T* resultScratch){
    
    int gid = get_global_id(0);
    int size = get_global_size(0);
    
    resultScratch[gid] = 0.0;
    for(int i = gid; i < num; i += size){
    	resultScratch[gid] += -log(prob_data[i * dim + static_cast<int>(label[i])]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    if(gid < 128)
    	resultScratch[gid] += resultScratch[gid + 128];
    barrier(CLK_LOCAL_MEM_FENCE);
    if(gid < 64)
    	resultScratch[gid] += resultScratch[gid + 64];
    if(gid < 32)
    	resultScratch[gid] += resultScratch[gid + 32];
    if(gid < 16)
    	resultScratch[gid] += resultScratch[gid + 16];
    if(gid < 8)
    	resultScratch[gid] += resultScratch[gid + 8];
    if(gid < 4)
    	resultScratch[gid] += resultScratch[gid + 4];
    if(gid < 2)
    	resultScratch[gid] += resultScratch[gid + 2];
    if(gid < 1){
    	resultScratch[gid] += resultScratch[gid + 1];
    	loss[0] = resultScratch[gid];
    }
}
template __attribute__ ((mangled_name(softmax_float))) __kernel void softmax (__global float* prob_data, __global float* loss, __global float* label, int num, int dim, __local float* resultScratch);
template __attribute__ ((mangled_name(softmax_double))) __kernel void softmax (__global double* prob_data, __global double* loss, __global double* label, int num, int dim, __local double* resultScratch);

template <class T>
__kernel void softmax_div (const int num, const int dim, __global T* scale, __global T* data){
        //printf("softmax_div\n");
        int index = get_global_id(0);
        int total = get_global_size(0);
        for(index; index < num*dim; index +=  total){
        int n = index / dim;
        data[index] /= scale[n];
        }
}

template __attribute__ ((mangled_name(softmax_div_float))) __kernel void softmax_div (const int num, const int dim, __global float* scale, __global float* data);
template __attribute__ ((mangled_name(softmax_div_double))) __kernel void softmax_div (const int num, const int dim, __global double* scale, __global double* data);
