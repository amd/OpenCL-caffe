template <class Dtype>
__kernel void CLLBackward(const int count, const int channels,
    const Dtype margin, const bool legacy_version, const Dtype alpha,
    __global const Dtype* y, __global const Dtype* diff, __global const Dtype* dist_sq,
    __global Dtype *bottom_diff) {
    int i = get_global_id(0);
    if(i < count) {
        int n = i / channels;  // the num index, to access y and dist_sq
        if (static_cast<int>(y[n])) {  // similar pairs
            bottom_diff[i] = alpha * diff[i];
        } else {  // dissimilar pairs
            Dtype mdist(0.0);
            Dtype beta(0.0);
            if (legacy_version) {
                mdist = (margin - dist_sq[n]);
                beta = -alpha;
            } else {
                Dtype dist = sqrt(dist_sq[n]);
                mdist = (margin - dist);
                beta = -alpha * mdist / (dist + Dtype(1e-4)) * diff[i];
            }
            if (mdist > 0.0) {
                bottom_diff[i] = beta;
            } else {
                bottom_diff[i] = 0;
            }
       }
   }
}

template __attribute__((mangled_name(CLLBackward_float))) __kernel void CLLBackward(const int count, const int channels,
    const float margin, const bool legacy_version, const float alpha,
    __global const float* y, __global const float* diff, __global const float* dist_sq,
    __global float *bottom_diff);
template __attribute__((mangled_name(CLLBackward_double))) __kernel void CLLBackward(const int count, const int channels,
    const double margin, const bool legacy_version, const double alpha,
    __global const double* y, __global const double* diff, __global const double* dist_sq,
    __global double *bottom_diff);
