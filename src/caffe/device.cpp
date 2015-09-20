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

#include "caffe/common.hpp"
#include "caffe/device.hpp"
#include <stdio.h>
#include <fstream>
#include <iostream>
#include <malloc.h>
#include <dirent.h>

namespace caffe {
#ifndef CPU_ONLY
string buildOption = "-x clc++ ";
std::string oclKernelPath = "./src/caffe/ocl/";
Device amdDevice;

Device::~Device() {
  ReleaseKernels();
  free((void*) platformIDs);
  free (DeviceIDs);
  clReleaseProgram (Program);
  clReleaseCommandQueue (CommandQueue);
  clReleaseCommandQueue (CommandQueue_helper);
  clReleaseContext (Context);
  LOG(INFO) << "device destructor";
}

cl_int Device::Init(int deviceId) {

  DisplayPlatformInfo();

  clGetPlatformIDs(0, NULL, &numPlatforms);
  cl_platform_id PlatformIDs[numPlatforms];
  clGetPlatformIDs(numPlatforms, PlatformIDs, NULL);

  size_t nameLen;
  cl_int res = clGetPlatformInfo(PlatformIDs[0], CL_PLATFORM_NAME, 64,
      platformName, &nameLen);
  if (res != CL_SUCCESS) {
    fprintf(stderr, "Err: Failed to Get Platform Info\n");
    return 0;
  }
  platformName[nameLen] = 0;

  GetDeviceInfo();
  cl_uint uiNumDevices;
  cl_bool unified_memory = false;
  clGetDeviceIDs(PlatformIDs[0], CL_DEVICE_TYPE_GPU, 0, NULL, &numDevices);
  uiNumDevices = numDevices;
  if (0 == uiNumDevices) {
    LOG(FATAL) << "Err: No GPU devices";
  } else {
    pDevices = (cl_device_id *) malloc(uiNumDevices * sizeof(cl_device_id));
    OCL_CHECK(
        clGetDeviceIDs(PlatformIDs[0], CL_DEVICE_TYPE_GPU, uiNumDevices,
            pDevices, &uiNumDevices));
    if (deviceId == -1) {
      int i;
      for (i = 0; i < (int) uiNumDevices; i++) {
        clGetDeviceInfo(pDevices[i], CL_DEVICE_HOST_UNIFIED_MEMORY,
            sizeof(cl_bool), &unified_memory, NULL);
        if (!unified_memory) { //skip iGPU
          //we pick the first dGPU we found
          pDevices[0] = pDevices[i];
          device_id = i;
          LOG(INFO) << "Picked default device type : dGPU " << device_id;
          break;
        }
      }
      if (i == uiNumDevices) {
        LOG(FATAL) << "Cannot find any dGPU! ";
      }
    } else if (deviceId >= 0 && deviceId < uiNumDevices) {
      pDevices[0] = pDevices[deviceId];
      device_id = deviceId;
      LOG(INFO) << "Picked device type : GPU " << device_id;
    } else {
      LOG(FATAL) << "  Invalid GPU deviceId! ";
    }
  }

  Context = clCreateContext(NULL, 1, pDevices, NULL, NULL, NULL);
  if (NULL == Context) {
    fprintf(stderr, "Err: Failed to Create Context\n");
    return 0;
  }
  CommandQueue = clCreateCommandQueue(Context, pDevices[0],
      CL_QUEUE_PROFILING_ENABLE, NULL);
  CommandQueue_helper = clCreateCommandQueue(Context, pDevices[0],
      CL_QUEUE_PROFILING_ENABLE, NULL);
  if (NULL == CommandQueue || NULL == CommandQueue_helper) {
    fprintf(stderr, "Err: Failed to Create Commandqueue\n");
    return 0;
  }
  BuildProgram (oclKernelPath);
  row = clblasRowMajor;
  col = clblasColumnMajor;
  return 0;
}

void Device::BuildProgram(std::string kernel_dir) {
  std::string strSource = "";
  DIR *ocl_dir;
  struct dirent *dirp;
  if ((ocl_dir = opendir(kernel_dir.c_str())) == NULL) {
    fprintf(stderr, "Err: Open ocl dir failed!\n");
  }
  while ((dirp = readdir(ocl_dir)) != NULL) {
    //Ignore hidden files
    if (dirp->d_name[0] == '.')
      continue;
    std::string file_name = std::string(dirp->d_name);
    //Skip non *.cl files
    size_t last_dot_pos = file_name.find_last_of(".");
    if (file_name.substr(last_dot_pos + 1) != "cl")
      continue;

    std::string ocl_kernel_full_path = kernel_dir + file_name;
    std::string tmpSource = "";
    ConvertToString(ocl_kernel_full_path.c_str(), tmpSource);
    strSource += tmpSource;
  }
  const char *pSource;
  pSource = strSource.c_str();
  size_t uiArrSourceSize[] = { 0 };
  uiArrSourceSize[0] = strlen(pSource);
  Program = NULL;
  Program = clCreateProgramWithSource(Context, 1, &pSource, uiArrSourceSize,
      NULL);
  if (NULL == Program) {
    fprintf(stderr, "Err: Failed to create program\n");
  }
  cl_int iStatus = clBuildProgram(Program, 1, pDevices, buildOption.c_str(),
      NULL, NULL);
  LOG(INFO) << "Build Program";
  if (CL_SUCCESS != iStatus) {
    fprintf(stderr, "Err: Failed to build program\n");
    char szBuildLog[16384];
    clGetProgramBuildInfo(Program, *pDevices, CL_PROGRAM_BUILD_LOG,
        sizeof(szBuildLog), szBuildLog, NULL);
    std::cout << szBuildLog;
    clReleaseProgram (Program);
  }
}

//Use to read OpenCL source code
cl_int Device::ConvertToString(std::string pFileName, std::string &Str) {
  size_t uiSize = 0;
  size_t uiFileSize = 0;
  char *pStr = NULL;
  char *tmp = (char*) pFileName.data();
  std::fstream fFile(tmp, (std::fstream::in | std::fstream::binary));
  if (fFile.is_open()) {
    fFile.seekg(0, std::fstream::end);
    uiSize = uiFileSize = (size_t) fFile.tellg();
    fFile.seekg(0, std::fstream::beg);
    pStr = new char[uiSize + 1];

    if (NULL == pStr) {
      fFile.close();
      return 0;
    }
    fFile.read(pStr, uiFileSize);
    fFile.close();
    pStr[uiSize] = '\0';
    Str = pStr;
    delete[] pStr;
    return 0;
  }
  LOG(ERROR) << "Err: Failed to open cl file!";
  return -1;
}

cl_kernel Device::GetKernel(std::string kernel_name) {
  std::map<std::string, cl_kernel>::iterator it = Kernels.find(kernel_name);
  if (it == Kernels.end()) {
    cl_int _err = 0;
    cl_kernel kernel = clCreateKernel(Program, kernel_name.c_str(), &_err);
    OCL_CHECK(_err);
    Kernels[kernel_name] = kernel;
  }
  return Kernels[kernel_name];
}

void Device::ReleaseKernels() {
  std::map<std::string, cl_kernel>::iterator it;
  for (it = Kernels.begin(); it != Kernels.end(); it++) {
    clReleaseKernel(it->second);
  }
}

void Device::DisplayPlatformInfo() {
  cl_int err;

  err = clGetPlatformIDs(0, NULL, &numPlatforms);
  if (err != CL_SUCCESS || numPlatforms <= 0) {
    LOG(ERROR) << "Failed to find any OpenCL platform.";
    return;
  }

  platformIDs = (cl_platform_id *) malloc(
      sizeof(cl_platform_id) * numPlatforms);
  err = clGetPlatformIDs(numPlatforms, platformIDs, NULL);
  if (err != CL_SUCCESS) {
    LOG(ERROR) << "Failed to find any OpenCL platform.";
    return;
  }

  LOG(INFO) << "Number of platforms found:" << numPlatforms;

  //iterate through the list of platforms displaying platform information
  for (cl_uint i = 0; i < numPlatforms; i++) {
    DisplayInfo(platformIDs[i], CL_PLATFORM_NAME, "CL_PLATFORM_NAME");
    DisplayInfo(platformIDs[i], CL_PLATFORM_PROFILE, "CL_PLATFORM_PROFILE");
    DisplayInfo(platformIDs[i], CL_PLATFORM_VERSION, "CL_PLATFORM_VERSION");
    DisplayInfo(platformIDs[i], CL_PLATFORM_VENDOR, "CL_PLATFORM_VENDOR");
    DisplayInfo(platformIDs[i], CL_PLATFORM_EXTENSIONS,
        "CL_PLATFORM_EXTENSIONS");
  }

}

void Device::DisplayInfo(cl_platform_id id, cl_platform_info name,
    std::string str) {
  cl_int err;
  std::size_t paramValueSize;

  err = clGetPlatformInfo(id, name, 0, NULL, &paramValueSize);
  if (err != CL_SUCCESS) {
    LOG(ERROR) << "Failed to find OpenCL platform:" << str;
    return;
  }

  char * info = (char *) alloca(sizeof(char) * paramValueSize);
  err = clGetPlatformInfo(id, name, paramValueSize, info, NULL);
  if (err != CL_SUCCESS) {
    LOG(ERROR) << "Failed to find OpenCL platform:" << str;
    return;
  }

  LOG(INFO) << "\t" << str << "\t" << info;
}

void Device::GetDeviceInfo() {
  cl_int err;
  //by default, we select the first platform. can be extended for more platforms
  //query GPU device for now
  err = clGetDeviceIDs(platformIDs[0], CL_DEVICE_TYPE_GPU, 0, NULL,
      &numDevices);
  // we allow program run if no GPU is found. Just return. No error reported.
  if (numDevices < 1) {
    LOG(INFO) << "No GPU Devices found for platform" << platformIDs[0];
    LOG(WARNING) << "No GPU Devices found for platform" << platformIDs[0];
    return;
  }

  DeviceIDs = (cl_device_id *) malloc(sizeof(cl_device_id) * numDevices);
  err = clGetDeviceIDs(platformIDs[0], CL_DEVICE_TYPE_GPU, numDevices,
      DeviceIDs, NULL);
  if (err != CL_SUCCESS) {
    LOG(INFO) << "Failed to find any GPU devices.";
    return;
  }

  LOG(INFO) << "Number of devices found:" << numDevices;
  for (cl_uint i = 0; i < numDevices; i++) {
    LOG(INFO) << "\t" << "DeviceID" << ":\t" << DeviceIDs[i];
    DisplayDeviceInfo < cl_device_type
        > (DeviceIDs[i], CL_DEVICE_TYPE, "Device Type");
    DisplayDeviceInfo < cl_bool
        > (DeviceIDs[i], CL_DEVICE_HOST_UNIFIED_MEMORY, "Is it integrated GPU?");
    DisplayDeviceInfo < cl_uint
        > (DeviceIDs[i], CL_DEVICE_MAX_CLOCK_FREQUENCY, "Max clock frequency MHz");
    DisplayDeviceInfo < cl_bool
        > (DeviceIDs[i], CL_DEVICE_HOST_UNIFIED_MEMORY, "Host-Device unified mem");
    DisplayDeviceInfo < cl_bool
        > (DeviceIDs[i], CL_DEVICE_ERROR_CORRECTION_SUPPORT, "ECC support");
    DisplayDeviceInfo < cl_bool
        > (DeviceIDs[i], CL_DEVICE_ENDIAN_LITTLE, "Endian little");
    DisplayDeviceInfo < cl_uint
        > (DeviceIDs[i], CL_DEVICE_MAX_COMPUTE_UNITS, "Max compute units");
    DisplayDeviceInfo < size_t
        > (DeviceIDs[i], CL_DEVICE_MAX_WORK_GROUP_SIZE, "Max work group size");
    DisplayDeviceInfo < cl_uint
        > (DeviceIDs[i], CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, "Max work item dimensions");
    DisplayDeviceInfo<size_t *>(DeviceIDs[i], CL_DEVICE_MAX_WORK_ITEM_SIZES,
        "Max work item sizes");
    DisplayDeviceInfo < cl_command_queue_properties
        > (DeviceIDs[i], CL_DEVICE_QUEUE_PROPERTIES, "CL_DEVICE_QUEUE_PROPERTIES");
    DisplayDeviceInfo < cl_device_exec_capabilities
        > (DeviceIDs[i], CL_DEVICE_EXECUTION_CAPABILITIES, "CL_DEVICE_EXECUTION_CAPABILITIES");
    DisplayDeviceInfo < cl_ulong
        > (DeviceIDs[i], CL_DEVICE_MAX_MEM_ALLOC_SIZE, "Max mem alloc size");
    DisplayDeviceInfo < cl_ulong
        > (DeviceIDs[i], CL_DEVICE_GLOBAL_MEM_SIZE, "Global mem size");
    DisplayDeviceInfo < cl_ulong
        > (DeviceIDs[i], CL_DEVICE_LOCAL_MEM_SIZE, "Local mem size");
  }

}

void Device::DeviceQuery() {
  DisplayPlatformInfo();

  clGetPlatformIDs(0, NULL, &numPlatforms);
  cl_platform_id PlatformIDs[numPlatforms];
  clGetPlatformIDs(numPlatforms, PlatformIDs, NULL);

  size_t nameLen;
  cl_int res = clGetPlatformInfo(PlatformIDs[0], CL_PLATFORM_NAME, 64,
      platformName, &nameLen);
  if (res != CL_SUCCESS) {
    fprintf(stderr, "Err: Failed to Get Platform Info\n");
    return;
  }
  platformName[nameLen] = 0;

  GetDeviceInfo();
}

template <typename T>
void Device::DisplayDeviceInfo(cl_device_id id, cl_device_info name,
    std::string str) {
  cl_int err;
  std::size_t paramValueSize;

  err = clGetDeviceInfo(id, name, 0, NULL, &paramValueSize);
  if (err != CL_SUCCESS) {
    LOG(ERROR) << "Failed to find OpenCL device info:" << str;
    return;
  }

  std::string content;
  T * info = (T *) alloca(sizeof(T) * paramValueSize);
  err = clGetDeviceInfo(id, name, paramValueSize, info, NULL);
  if (err != CL_SUCCESS) {
    LOG(ERROR) << "Failed to find OpenCL device info:" << str;
    return;
  }

  switch (name) {
  case CL_DEVICE_TYPE: {
    std::string deviceType;
    appendBitfield < cl_device_type
        > (*(reinterpret_cast<cl_device_type*>(info)), CL_DEVICE_TYPE_CPU, "CL_DEVICE_TYPE_CPU", deviceType);

    appendBitfield < cl_device_type
        > (*(reinterpret_cast<cl_device_type*>(info)), CL_DEVICE_TYPE_GPU, "CL_DEVICE_TYPE_GPU", deviceType);

    appendBitfield < cl_device_type
        > (*(reinterpret_cast<cl_device_type*>(info)), CL_DEVICE_TYPE_ACCELERATOR, "CL_DEVICE_TYPE_ACCELERATOR", deviceType);

    appendBitfield < cl_device_type
        > (*(reinterpret_cast<cl_device_type*>(info)), CL_DEVICE_TYPE_DEFAULT, "CL_DEVICE_TYPE_DEFAULT", deviceType);

    LOG(INFO) << "\t " << str << ":\t" << deviceType;
  }
    break;
  case CL_DEVICE_EXECUTION_CAPABILITIES: {
    std::string memType;
    appendBitfield < cl_device_exec_capabilities
        > (*(reinterpret_cast<cl_device_exec_capabilities*>(info)), CL_EXEC_KERNEL, "CL_EXEC_KERNEL", memType);

    appendBitfield < cl_device_exec_capabilities
        > (*(reinterpret_cast<cl_device_exec_capabilities*>(info)), CL_EXEC_NATIVE_KERNEL, "CL_EXEC_NATIVE_KERNEL", memType);

    LOG(INFO) << "\t " << str << ":\t" << memType;

  }
    break;
  case CL_DEVICE_QUEUE_PROPERTIES: {
    std::string memType;
    appendBitfield < cl_device_exec_capabilities
        > (*(reinterpret_cast<cl_device_exec_capabilities*>(info)), CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, "CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE", memType);

    appendBitfield < cl_device_exec_capabilities
        > (*(reinterpret_cast<cl_device_exec_capabilities*>(info)), CL_QUEUE_PROFILING_ENABLE, "CL_QUEUE_PROFILING_ENABLE", memType);

    LOG(INFO) << "\t " << str << ":\t" << memType;
  }
    break;
  default:
    LOG(INFO) << "\t" << str << ":\t" << *info;
    break;
  }

}

template <typename T>
void Device::appendBitfield(T info, T value, std::string name,
    std::string &str) {
  if (info & value) {
    if (str.length() > 0) {
      str.append(" | ");
    }
    str.append(name);
  }
}

#endif
}  // namespace caffe

