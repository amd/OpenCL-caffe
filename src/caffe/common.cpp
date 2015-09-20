#include <glog/logging.h>
#include <cstdio>
#include <ctime>

#include "caffe/common.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

shared_ptr<Caffe> Caffe::singleton_;

// random seeding
int64_t cluster_seedgen(void) {
  //To fix: for now we use fixed seed to get same result each time
   int64_t s, seed, pid;
   FILE* f = fopen("/dev/urandom", "rb");
   if (f && fread(&seed, 1, sizeof(seed), f) == sizeof(seed)) {
   fclose(f);
   return seed;
   }

   LOG(INFO) << "System entropy source not available, "
   "using fallback algorithm to generate seed instead.";
   if (f)
   fclose(f);

   pid = getpid();
   s = time(NULL);
   seed = abs(((s * 181) * ((pid - 83) * 359)) % 104729);
   return seed;
}

void GlobalInit(int* pargc, char*** pargv) {
  // Google flags.
  ::gflags::ParseCommandLineFlags(pargc, pargv, true);
  // Google logging.
  ::google::InitGoogleLogging(*(pargv)[0]);
  // Provide a backtrace on segfault.
  ::google::InstallFailureSignalHandler();
}

#ifdef CPU_ONLY  // CPU-only Caffe.

Caffe::Caffe()
: random_generator_(), mode_(Caffe::CPU) {
}

Caffe::~Caffe() {
}

void Caffe::set_random_seed(const unsigned int seed) {
  // RNG seed
  Get().random_generator_.reset(new RNG(seed));
}

void Caffe::SetDevice(const int device_id) {
  NO_GPU;
}

void Caffe::DeviceQuery() {
  NO_GPU;
}

class Caffe::RNG::Generator {
  public:
  Generator() : rng_(new caffe::rng_t(cluster_seedgen())) {}
  explicit Generator(unsigned int seed) : rng_(new caffe::rng_t(seed)) {}
  caffe::rng_t* rng() {return rng_.get();}
  private:
  shared_ptr<caffe::rng_t> rng_;
};

Caffe::RNG::RNG() : generator_(new Generator()) {}

Caffe::RNG::RNG(unsigned int seed) : generator_(new Generator(seed)) {}

Caffe::RNG& Caffe::RNG::operator=(const RNG& other) {
  generator_ = other.generator_;
  return *this;
}

void* Caffe::RNG::generator() {
  return static_cast<void*>(generator_->rng());
}

#else  // Normal GPU + CPU Caffe.

Caffe::Caffe() {
  amdDevice.Init();
  cl_int err = clblasSetup();
  if (err != CL_SUCCESS) {
    LOG(ERROR) << "clBLAS setup failed " << err;
  }
}

Caffe::~Caffe() {
  clblasTeardown();
}

void Caffe::set_random_seed(const unsigned int seed) {
	// RNG seed
	Get().random_generator_.reset(new RNG(seed));
        caffe_gpu_uniform(0, NULL, seed);
        caffe_gpu_uniform((float*)NULL, 0, (float)0.0, (float)1.0, seed);
}

void Caffe::SetDevice(const int device_id) {
  if (amdDevice.GetDevice() == device_id) {
    return;
  }
  amdDevice.Init(device_id);
}

void Caffe::DeviceQuery() {
  amdDevice.DeviceQuery();
}

class Caffe::RNG::Generator {
  public:
    Generator()
        : rng_(new caffe::rng_t(cluster_seedgen())) {
    }
    explicit Generator(unsigned int seed)
        : rng_(new caffe::rng_t(seed)) {
    }
    caffe::rng_t* rng() {
      return rng_.get();
    }
  private:
    shared_ptr<caffe::rng_t> rng_;
};

Caffe::RNG::RNG()
    : generator_(new Generator()) {
}

Caffe::RNG::RNG(unsigned int seed)
    : generator_(new Generator(seed)) {
}

Caffe::RNG& Caffe::RNG::operator=(const RNG& other) {
  generator_.reset(other.generator_.get());
  return *this;
}

void* Caffe::RNG::generator() {
  return static_cast<void*>(generator_->rng());
}

#endif  // CPU_ONLY

}  // namespace caffe
