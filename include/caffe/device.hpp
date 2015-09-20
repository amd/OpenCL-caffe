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

#ifndef CAFFE_DEVICE_HPP
#define CAFFE_DEVICE_HPP
#include <string>
#include <fstream>
#include "caffe/common.hpp"
namespace caffe {
#ifndef CPU_ONLY
class Device {
  public:
    Device()
        : numPlatforms(0), numDevices(0), device_id(INT_MIN) {
    }
    ~Device();
    cl_uint numPlatforms;
    cl_platform_id * platformIDs;
    char platformName[64];
    char openclVersion[64];
    cl_uint numDevices;
    cl_device_id * DeviceIDs;

    cl_context Context;
    cl_command_queue CommandQueue;
    cl_command_queue CommandQueue_helper;
    cl_program Program;
    cl_device_id * pDevices;
    int device_id;

    clblasOrder col;
    clblasOrder row;
    std::map<std::string, cl_kernel> Kernels;

    cl_int Init(int device_id = -1);
    cl_int ConvertToString(std::string pFileName, std::string &Str);
    void DisplayPlatformInfo();
    void DisplayInfo(cl_platform_id id, cl_platform_info name, std::string str);

    void GetDeviceInfo();
    void DeviceQuery();
    int GetDevice() {
      return device_id;
    }
    ;
    void BuildProgram(std::string kernel_dir);

    template <typename T>
    void DisplayDeviceInfo(cl_device_id id, cl_device_info name,
        std::string str);
    template <typename T>
    void appendBitfield(T info, T value, std::string name, std::string &str);

    cl_kernel GetKernel(std::string kernel_name);
    void ReleaseKernels();
};
extern std::string buildOption;
extern Device amdDevice;
#endif
}  // namespace caffe

#endif //CAFFE_DEVICE_HPP

