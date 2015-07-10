#ifndef CAFFE_DEVICE_HPP
#define CAFFE_DEVICE_HPP
#include <CL/cl.h>
#include <string>
#include <fstream>
#include "caffe/common.hpp"
namespace caffe {

class Device{
public:
    Device():numPlatforms(0),numDevices(0){}
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
    clblasOrder col;
    clblasOrder row;

     
    cl_int Init(); 
    cl_int ConvertToString(const char *pFileName,std::string &Str);
    void DisplayPlatformInfo();
    void DisplayInfo(cl_platform_id id, cl_platform_info name, std::string str);

    void GetDeviceInfo();

    template <typename T>
    void DisplayDeviceInfo(cl_device_id id, cl_device_info name, std::string str);
    template <typename T>
    void appendBitfield(T info, T value, std::string name, std::string &str);
    

};
extern char* buildOption;
extern Device amdDevice;

}  // namespace caffe

#endif //CAFFE_DEVICE_HPP

