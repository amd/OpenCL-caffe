// The main caffe test code. Your test cpp code should include this hpp
// to allow a main function to be compiled into the binary.

#include "caffe/caffe.hpp"
#include "caffe/common.hpp"
#include "caffe/test/test_caffe_main.hpp"


int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  caffe::GlobalInit(&argc, &argv);
#ifndef CPU_ONLY
  // Before starting testing, let's first print out a few cuda defice info.
  int device = 0;
  if (argc > 1) {
    // Use the given device
    device = atoi(argv[1]);
    caffe::amdDevice.Init(device);
    cout << "Setting to use device " << device << endl;
  } else if (OPENCL_TEST_DEVICE >= 0) {
    // Use the device assigned in build configuration; but with a lower priority
    device = OPENCL_TEST_DEVICE;
  }
  cout << "Current device id: " << device << endl;
  caffe::amdDevice.Init();
#endif
  // invoke the test.
  return RUN_ALL_TESTS();
}
