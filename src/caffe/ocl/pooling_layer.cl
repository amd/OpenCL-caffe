/*************************************************************************************
 * Copyright (c) 2015, Advanced Micro Devices, Inc.  
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this 
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice, 
 * this list of conditions and the following disclaimer in the documentation and/or
 *  other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, 
 * INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, 
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
 * OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 **************************************************************************************/

template <class T>
__kernel void MaxPoolForward(const int nthreads, __global T* bottom_data, const int num, const int channels, const int height, const int width, const int pooled_height, const int pooled_width, const int kernel_h, const int kernel_w, const int stride_h, const int stride_w, const int pad_h, const int pad_w, __global T* top_data, __global int* mask, __global T* top_mask) {
  int index = get_global_id(0);
  int tmp = get_global_size(0);
  for(index; index < nthreads; index += tmp) {
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;
    int hstart = ph * stride_h - pad_h;
    int wstart = pw * stride_w - pad_w;
    const int hend = min(hstart + kernel_h, height);
    const int wend = min(wstart + kernel_w, width);
    hstart = max(hstart, 0);
    wstart = max(wstart, 0);
    T maxval = -FLT_MAX;
    int maxidx = -1;
    bottom_data =
    bottom_data + (n * channels + c) * height * width;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        if (bottom_data[h * width + w] > maxval) {
          maxidx = h * width + w;
          maxval = bottom_data[maxidx];
        }
      }
    }
    top_data[index] = maxval;
    if (mask) {
      mask[index] = maxidx;
    } else {
      top_mask[index] = maxidx;
    }
  }
}
template __attribute__((mangled_name(MaxPoolForward_float))) __kernel void MaxPoolForward(const int nthreads, __global float* bottom_data, const int num, const int channels, const int height, const int width, const int pooled_height, const int pooled_width,const int kernel_h, const int kernel_w, const int stride_h, const int stride_w, const int pad_h, const int pad_w, __global float* top_data, __global int* mask, __global float* top_mask);
template __attribute__((mangled_name(MaxPoolForward_double))) __kernel void MaxPoolForward(const int nthreads, __global double* bottom_data, const int num, const int channels, const int height, const int width, const int pooled_height, const int pooled_width, const int kernel_h, const int kernel_w, const int stride_h, const int stride_w, const int pad_h, const int pad_w, __global double* top_data, __global int* mask, __global double* top_mask);

template <class T>
__kernel void AvePoolForward(const int nthreads, __global T* bottom_data, const int num, const int channels, const int height, const int width, const int pooled_height, const int pooled_width, const int kernel_h, const int kernel_w, const int stride_h, const int stride_w, const int pad_h, const int pad_w,__global T* top_data) {
  int index = get_global_id(0);
  int tmp = get_global_size(0);
  for(index; index < nthreads; index+=tmp) {
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels; int hstart = ph * stride_h - pad_h; int wstart = pw * stride_w - pad_w;
    int hend = min(hstart + kernel_h, height + pad_h);
    int wend = min(wstart + kernel_w, width + pad_w);
    const int pool_size = (hend - hstart) * (wend - wstart);
    hstart = max(hstart, 0);
    wstart = max(wstart, 0);
    hend = min(hend, height);
    wend = min(wend, width);
    T aveval = 0;
    bottom_data =
    bottom_data + (n * channels + c) * height * width;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        aveval += bottom_data[h * width + w];
      }
    }
    top_data[index] = aveval / pool_size;
  }

}
template __attribute__((mangled_name(AvePoolForward_float))) __kernel void AvePoolForward(const int nthreads, __global float* bottom_data, const int num, const int channels, const int height, const int width, const int pooled_height, const int pooled_width, const int kernel_h, const int kernel_w, const int stride_h, const int stride_w, const int pad_h, const int pad_w,__global float* top_data);
template __attribute__((mangled_name(AvePoolForward_double))) __kernel void AvePoolForward(const int nthreads, __global double* bottom_data, const int num, const int channels, const int height, const int width, const int pooled_height, const int pooled_width, const int kernel_h, const int kernel_w, const int stride_h, const int stride_w, const int pad_h, const int pad_w,__global double* top_data);

template <class T>
__kernel void StoPoolForwardTrain(const int nthreads, __global T* bottom_data, const int clnum, const int channels, const int height, const int width, const int pooled_height, const int pooled_width, const int kernel_h, const int kernel_w, const int stride_h, const int stride_w, __global T* rand_idx, __global T* top_data) {
  int index = get_global_id(0);
  int tmp = get_global_size(0);
  for(index; index < nthreads; index+=tmp) {
    const int pw = index % pooled_width;
    const int ph = (index / pooled_width) % pooled_height;
    const int c = (index / pooled_width / pooled_height) % channels;
    const int n = index / pooled_width / pooled_height / channels;
    const int hstart = ph * stride_h;
    const int hend = min(hstart + kernel_h, height);
    const int wstart = pw * stride_w;
    const int wend = min(wstart + kernel_w, width);
    T cumsum = 0.;
    bottom_data = bottom_data + (n * channels + c) * height * width;
    // First pass: get sum
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        cumsum += bottom_data[h * width + w];
      }
    }
    const float thres = rand_idx[index] * cumsum;
    // Second pass: get value, and set index.
    cumsum = 0;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        cumsum += bottom_data[h * width + w];
        if (cumsum >= thres) {
          rand_idx[index] = ((n * channels + c) * height + h) * width + w;
          top_data[index] = bottom_data[h * width + w];
          return;
        }
      }
    }
  }
}
template __attribute__((mangled_name(StoPoolForwardTrain_float))) __kernel void StoPoolForwardTrain(const int nthreads, __global float* bottom_data, const int num, const int channels, const int height, const int width, const int pooled_height, const int pooled_width, const int kernel_h, const int kernel_w, const int stride_h, const int stride_w, __global float* idx_data, __global float* top_data);
template __attribute__((mangled_name(StoPoolForwardTrain_double))) __kernel void StoPoolForwardTrain(const int nthreads, __global double* bottom_data, const int num, const int channels, const int height, const int width, const int pooled_height, const int pooled_width, const int kernel_h, const int kernel_w, const int stride_h, const int stride_w, __global double* idx_data, __global double* top_data);

template <class T>
__kernel void StoPoolForwardTest(const int count, __global T* bottom_data, const int clnum, const int channels, const int height, const int width, const int pooled_height, const int pooled_width, const int kernel_h, const int kernel_w, const int stride_h, const int stride_w, __global T* top_data) {
  int index = get_global_id(0);
  int tmp = get_global_size(0);
  for(index; index < count; index+=tmp) {
    const int pw = index % pooled_width;
    const int ph = (index / pooled_width) % pooled_height;
    const int c = (index / pooled_width / pooled_height) % channels;
    const int n = index / pooled_width / pooled_height / channels;
    const int hstart = ph * stride_h;
    const int hend = min(hstart + kernel_h, height);
    const int wstart = pw * stride_w;
    const int wend = min(wstart + kernel_w, width);
    // We set cumsum to be 0 to avoid divide-by-zero problems    T cumsum = FLT_MIN;
    T cumsum = FLT_MIN;
    T cumvalues = 0.;
    bottom_data = bottom_data + (n * channels + c) * height * width;
    // First pass: get sum
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        cumsum += bottom_data[h * width + w];
        cumvalues += bottom_data[h * width + w] * bottom_data[h * width + w];
      }
    }
    top_data[index] = cumvalues / cumsum;}
}
template __attribute__((mangled_name(StoPoolForwardTest_float))) __kernel void StoPoolForwardTest(const int nthreads, __global float* bottom_data, const int num, const int channels, const int height, const int width, const int pooled_height, const int pooled_width, const int kernel_h, const int kernel_w, const int stride_h, const int stride_w,__global float* top_data);
template __attribute__((mangled_name(StoPoolForwardTest_double))) __kernel void StoPoolForwardTest(const int nthreads, __global double* bottom_data, const int num, const int channels, const int height, const int width, const int pooled_height, const int pooled_width, const int kernel_h, const int kernel_w, const int stride_h, const int stride_w, __global double* top_data);

template <class T>
__kernel void MaxPoolBackward(const int nthreads, __global T* top_diff,
    __global int* mask, __global T* top_mask, const int num,
    const int channels, const int height, const int width,
    const int pooled_height, const int pooled_width, const int kernel_h,
    const int kernel_w, const int stride_h, const int stride_w, const int pad_h,
    const int pad_w, __global T* const bottom_diff) {
  int index = get_global_id(0);
  int total = get_global_size(0);
  for(index; index < nthreads; index += total) {
    // find out the local index
    // find out the local offset
    const int w = index % width;
    const int h = (index / width) % height;
    const int c = (index / width / height) % channels;
    const int n = index / width / height / channels;
    const int phstart =
    (h + pad_h < kernel_h) ? 0 : (h + pad_h - kernel_h) / stride_h + 1;
    const int phend = min((h + pad_h) / stride_h + 1, pooled_height);
    const int pwstart =
    (w + pad_w < kernel_w) ? 0 : (w + pad_w - kernel_w) / stride_w + 1;
    const int pwend = min((w + pad_w) / stride_w + 1, pooled_width);
    T gradient = 0;
    const int offset = (n * channels + c) * pooled_height * pooled_width;
    top_diff += offset;
    if (mask) {
      mask = mask + offset;
      for (int ph = phstart; ph < phend; ++ph) {
        for (int pw = pwstart; pw < pwend; ++pw) {
          if (mask[ph * pooled_width + pw] == h * width + w) {
            gradient += top_diff[ph * pooled_width + pw];
          }
        }
      }
    } else {
      top_mask = top_mask + offset;
      for (int ph = phstart; ph < phend; ++ph) {
        for (int pw = pwstart; pw < pwend; ++pw) {
          if (top_mask[ph * pooled_width + pw] == h * width + w) {
            gradient += top_diff[ph * pooled_width + pw];
          }
        }
      }
    }
    bottom_diff[index] = gradient;
  }
}
template __attribute__((mangled_name(MaxPoolBackward_float))) __kernel void MaxPoolBackward(const int nthreads, __global float* const top_diff, __global int* const mask, __global float* const top_mask, const int num, const int channels, const int height, const int width, const int pooled_height, const int pooled_width, const int kernel_h, const int kernel_w, const int stride_h, const int stride_w, const int pad_h, const int pad_w, __global float* bottom_diff);
template __attribute__((mangled_name(MaxPoolBackward_double))) __kernel void MaxPoolBackward(const int nthreads, __global double* top_diff, __global int* const mask, __global double* const top_mask, const int num, const int channels, const int height, const int width, const int pooled_height, const int pooled_width, const int kernel_h, const int kernel_w, const int stride_h, const int stride_w, const int pad_h, const int pad_w, __global double* bottom_diff);

template <class T>
__kernel void AvePoolBackward(const int nthreads, __global T* top_diff, const int num, const int channels, const int height, const int width, const int pooled_height, const int pooled_width, const int kernel_h, const int kernel_w, const int stride_h, const int stride_w, const int pad_h, const int pad_w, __global T* const bottom_diff) {
  int index = get_global_id(0);
  int total = get_global_size(0);
  for(index; index < nthreads; index += total) {
    int w = index % width + pad_w;
    int h = (index / width) % height + pad_h;
    int c = (index / width / height) % channels;
    int n = index / width / height / channels;
    const int phstart = (h < kernel_h) ? 0 : (h - kernel_h) / stride_h + 1;
    const int phend = min(h / stride_h + 1, pooled_height);
    const int pwstart = (w < kernel_w) ? 0 : (w - kernel_w) / stride_w + 1;
    const int pwend = min(w / stride_w + 1, pooled_width);
    T gradient = 0;
    top_diff += (n * channels + c) * pooled_height * pooled_width;
    for (int ph = phstart; ph < phend; ++ph) {
      for (int pw = pwstart; pw < pwend; ++pw) {
        // figure out the pooling size
        int hstart = ph * stride_h - pad_h;
        int wstart = pw * stride_w - pad_w;
        int hend = min(hstart + kernel_h, height + pad_h);
        int wend = min(wstart + kernel_w, width + pad_w);
        int pool_size = (hend - hstart) * (wend - wstart);
        gradient += top_diff[ph * pooled_width + pw] / pool_size;
      }
    }
    bottom_diff[index] = gradient;
  }
}

template __attribute__((mangled_name(AvePoolBackward_float))) __kernel void AvePoolBackward(const int nthreads, __global float* top_diff, const int num, const int channels, const int height, const int width, const int pooled_height, const int pooled_width, const int kernel_h, const int kernel_w, const int stride_h, const int stride_w, const int pad_h, const int pad_w, __global float* bottom_diff);
template __attribute__((mangled_name(AvePoolBackward_double))) __kernel void AvePoolBackward(const int nthreads, __global double* top_diff, const int num, const int channels, const int height, const int width, const int pooled_height, const int pooled_width, const int kernel_h, const int kernel_w, const int stride_h, const int stride_w, const int pad_h, const int pad_w, __global double* bottom_diff);

template <class Dtype>
__kernel void StoPoolBackward(const int nthreads,
    __global Dtype* rand_idx, __global Dtype* top_diff,
    const int num, const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    const int kernel_h, const int kernel_w, const int stride_h,
    const int stride_w, __global Dtype* bottom_diff) {
  int index = get_global_id(0);
  int total = get_global_size(0);
  for(index; index < nthreads; index += total) {
    // find out the local index
    // find out the local offset
    const int w = index % width;
    const int h = (index / width) % height;
    const int c = (index / width / height) % channels;
    const int n = index / width / height / channels;
    const int phstart = (h < kernel_h) ? 0 : (h - kernel_h) / stride_h + 1;
    const int phend = min(h / stride_h + 1, pooled_height);
    const int pwstart = (w < kernel_w) ? 0 : (w - kernel_w) / stride_w + 1;
    const int pwend = min(w / stride_w + 1, pooled_width);
    Dtype gradient = 0;
    rand_idx =
    rand_idx + (n * channels + c) * pooled_height * pooled_width;
    top_diff =
    top_diff + (n * channels + c) * pooled_height * pooled_width;
    for (int ph = phstart; ph < phend; ++ph) {
      for (int pw = pwstart; pw < pwend; ++pw) {
        gradient += top_diff[ph * pooled_width + pw] *
        (index == static_cast<int>(rand_idx[ph * pooled_width + pw]));
      }
    }
    bottom_diff[index] = gradient;

  }
}
template __attribute__ ((mangled_name(StoPoolBackward_float))) __kernel void StoPoolBackward(const int nthreads,
    __global float* rand_idx, __global float* top_diff,
    const int num, const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    const int kernel_h, const int kernel_w, const int stride_h,
    const int stride_w, __global float* bottom_diff);
template __attribute__ ((mangled_name(StoPoolBackward_double))) __kernel void StoPoolBackward(const int nthreads,
    __global double* rand_idx, __global double* top_diff,
    const int num, const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    const int kernel_h, const int kernel_w, const int stride_h,
    const int stride_w, __global double* bottom_diff);
