#pragma OPENCL EXTENSION cl_amd_printf : enable

//beginning of the looooooong gpu_random_generator kernel 
//we use the open sourced threefry's GPU implementation
typedef uint uint32_t;

struct r123array4x32 {	uint32_t v[4]; };

enum r123_enum_threefry32x4 
{
	R_32x4_0_0 = 10, R_32x4_0_1 = 26,
	R_32x4_1_0 = 11, R_32x4_1_1 = 21,
	R_32x4_2_0 = 13, R_32x4_2_1 = 27,
	R_32x4_3_0 = 23, R_32x4_3_1 =  5,
	R_32x4_4_0 =  6, R_32x4_4_1 = 20,
	R_32x4_5_0 = 17, R_32x4_5_1 = 11,
	R_32x4_6_0 = 25, R_32x4_6_1 = 10,
	R_32x4_7_0 = 18, R_32x4_7_1 = 20
};

inline uint32_t	RotL_32(uint32_t x, unsigned int N)__attribute__((always_inline));
inline uint32_t RotL_32(uint32_t x, unsigned int N)
{
	return (x << (N & 31)) | (x >> ((32 - N) & 31));
}

typedef struct r123array4x32 threefry4x32_ctr_t;
typedef struct r123array4x32 threefry4x32_key_t;
typedef struct r123array4x32 threefry4x32_ukey_t;

inline threefry4x32_ctr_t threefry4x32_R(unsigned int Nrounds, threefry4x32_ctr_t in, threefry4x32_key_t k)__attribute__((always_inline));
inline threefry4x32_ctr_t threefry4x32_R(unsigned int Nrounds, threefry4x32_ctr_t in, threefry4x32_key_t k)
{
	threefry4x32_ctr_t	X;
	uint32_t			ks[4 + 1];
	int					i;
	ks[4] = 0x1BD11BDA;
	/*
	for (i = 0; i < 4; i++)
	{
		ks[i] = k.v[i];
		X.v[i] = in.v[i];
		ks[4] ^= k.v[i];
	}*/ 
	{
		ks[0] = k.v[0];
		X.v[0] = in.v[0];
		ks[4] ^= k.v[0];

		ks[1] = k.v[1];
		X.v[1] = in.v[1];
		ks[4] ^= k.v[1];

		ks[2] = k.v[2];
		X.v[2] = in.v[2];
		ks[4] ^= k.v[2];

		ks[3] = k.v[3];
		X.v[3] = in.v[3];
		ks[4] ^= k.v[3];
	}
	X.v[0] += ks[0];
	X.v[1] += ks[1];
	X.v[2] += ks[2];
	X.v[3] += ks[3];
	if (Nrounds > 0) 
	{
		X.v[0] += X.v[1];
		X.v[1] = RotL_32(X.v[1], R_32x4_0_0);
		X.v[1] ^= X.v[0];
		X.v[2] += X.v[3];
		X.v[3] = RotL_32(X.v[3], R_32x4_0_1);
		X.v[3] ^= X.v[2];
	} if (Nrounds > 1) {
		X.v[0] += X.v[3];
		X.v[3] = RotL_32(X.v[3], R_32x4_1_0);
		X.v[3] ^= X.v[0];
		X.v[2] += X.v[1];
		X.v[1] = RotL_32(X.v[1], R_32x4_1_1);
		X.v[1] ^= X.v[2];
	} if (Nrounds > 2) {
		X.v[0] += X.v[1];
		X.v[1] = RotL_32(X.v[1], R_32x4_2_0);
		X.v[1] ^= X.v[0];
		X.v[2] += X.v[3];
		X.v[3] = RotL_32(X.v[3], R_32x4_2_1);
		X.v[3] ^= X.v[2];
	} if (Nrounds > 3) {
		X.v[0] += X.v[3];
		X.v[3] = RotL_32(X.v[3], R_32x4_3_0);
		X.v[3] ^= X.v[0];
		X.v[2] += X.v[1];
		X.v[1] = RotL_32(X.v[1], R_32x4_3_1);
		X.v[1] ^= X.v[2];
	} if (Nrounds > 3) {
		X.v[0] += ks[1];
		X.v[1] += ks[2];
		X.v[2] += ks[3];
		X.v[3] += ks[4];
		X.v[4 - 1] += 1;
	} if (Nrounds > 4) {
		X.v[0] += X.v[1];
		X.v[1] = RotL_32(X.v[1], R_32x4_4_0);
		X.v[1] ^= X.v[0];
		X.v[2] += X.v[3];
		X.v[3] = RotL_32(X.v[3], R_32x4_4_1);
		X.v[3] ^= X.v[2];
	} if (Nrounds > 5) {
		X.v[0] += X.v[3];
		X.v[3] = RotL_32(X.v[3], R_32x4_5_0);
		X.v[3] ^= X.v[0];
		X.v[2] += X.v[1];
		X.v[1] = RotL_32(X.v[1], R_32x4_5_1);
		X.v[1] ^= X.v[2];
	} if (Nrounds > 6) {
		X.v[0] += X.v[1];
		X.v[1] = RotL_32(X.v[1], R_32x4_6_0);
		X.v[1] ^= X.v[0];
		X.v[2] += X.v[3];
		X.v[3] = RotL_32(X.v[3], R_32x4_6_1);
		X.v[3] ^= X.v[2];
	} if (Nrounds > 7) {
		X.v[0] += X.v[3];
		X.v[3] = RotL_32(X.v[3], R_32x4_7_0);
		X.v[3] ^= X.v[0];
		X.v[2] += X.v[1];
		X.v[1] = RotL_32(X.v[1], R_32x4_7_1);
		X.v[1] ^= X.v[2];
	} if (Nrounds > 7) {
		X.v[0] += ks[2];
		X.v[1] += ks[3];
		X.v[2] += ks[4];
		X.v[3] += ks[0];
		X.v[4 - 1] += 2;
	} if (Nrounds > 8) {
		X.v[0] += X.v[1];
		X.v[1] = RotL_32(X.v[1], R_32x4_0_0);
		X.v[1] ^= X.v[0];
		X.v[2] += X.v[3];
		X.v[3] = RotL_32(X.v[3], R_32x4_0_1);
		X.v[3] ^= X.v[2];
	} if (Nrounds > 9) {
		X.v[0] += X.v[3];
		X.v[3] = RotL_32(X.v[3], R_32x4_1_0);
		X.v[3] ^= X.v[0];
		X.v[2] += X.v[1];
		X.v[1] = RotL_32(X.v[1], R_32x4_1_1);
		X.v[1] ^= X.v[2];
	} if (Nrounds > 10) {
		X.v[0] += X.v[1];
		X.v[1] = RotL_32(X.v[1], R_32x4_2_0);
		X.v[1] ^= X.v[0];
		X.v[2] += X.v[3];
		X.v[3] = RotL_32(X.v[3], R_32x4_2_1);
		X.v[3] ^= X.v[2];
	} if (Nrounds > 11) {
		X.v[0] += X.v[3];
		X.v[3] = RotL_32(X.v[3], R_32x4_3_0);
		X.v[3] ^= X.v[0];
		X.v[2] += X.v[1];
		X.v[1] = RotL_32(X.v[1], R_32x4_3_1);
		X.v[1] ^= X.v[2];
	} if (Nrounds > 11) {
		X.v[0] += ks[3];
		X.v[1] += ks[4];
		X.v[2] += ks[0];
		X.v[3] += ks[1];
		X.v[4 - 1] += 3;
	} if (Nrounds > 12) {
		X.v[0] += X.v[1];
		X.v[1] = RotL_32(X.v[1], R_32x4_4_0);
		X.v[1] ^= X.v[0];
		X.v[2] += X.v[3];
		X.v[3] = RotL_32(X.v[3], R_32x4_4_1);
		X.v[3] ^= X.v[2];
	} if (Nrounds > 13) {
		X.v[0] += X.v[3];
		X.v[3] = RotL_32(X.v[3], R_32x4_5_0);
		X.v[3] ^= X.v[0];
		X.v[2] += X.v[1];
		X.v[1] = RotL_32(X.v[1], R_32x4_5_1);
		X.v[1] ^= X.v[2];
	} if (Nrounds > 14) {
		X.v[0] += X.v[1];
		X.v[1] = RotL_32(X.v[1], R_32x4_6_0);
		X.v[1] ^= X.v[0];
		X.v[2] += X.v[3];
		X.v[3] = RotL_32(X.v[3], R_32x4_6_1);
		X.v[3] ^= X.v[2];
	} if (Nrounds > 15) {
		X.v[0] += X.v[3];
		X.v[3] = RotL_32(X.v[3], R_32x4_7_0);
		X.v[3] ^= X.v[0];
		X.v[2] += X.v[1];
		X.v[1] = RotL_32(X.v[1], R_32x4_7_1);
		X.v[1] ^= X.v[2];
	} if (Nrounds > 15) {
		X.v[0] += ks[4];
		X.v[1] += ks[0];
		X.v[2] += ks[1];
		X.v[3] += ks[2];
		X.v[4 - 1] += 4;
	} if (Nrounds > 16) {
		X.v[0] += X.v[1];
		X.v[1] = RotL_32(X.v[1], R_32x4_0_0);
		X.v[1] ^= X.v[0];
		X.v[2] += X.v[3];
		X.v[3] = RotL_32(X.v[3], R_32x4_0_1);
		X.v[3] ^= X.v[2];
	} if (Nrounds > 17) {
		X.v[0] += X.v[3];
		X.v[3] = RotL_32(X.v[3], R_32x4_1_0);
		X.v[3] ^= X.v[0];
		X.v[2] += X.v[1];
		X.v[1] = RotL_32(X.v[1], R_32x4_1_1);
		X.v[1] ^= X.v[2];
	} if (Nrounds > 18) {
		X.v[0] += X.v[1];
		X.v[1] = RotL_32(X.v[1], R_32x4_2_0);
		X.v[1] ^= X.v[0];
		X.v[2] += X.v[3];
		X.v[3] = RotL_32(X.v[3], R_32x4_2_1);
		X.v[3] ^= X.v[2];
	} if (Nrounds > 19) {
		X.v[0] += X.v[3];
		X.v[3] = RotL_32(X.v[3], R_32x4_3_0);
		X.v[3] ^= X.v[0];
		X.v[2] += X.v[1];
		X.v[1] = RotL_32(X.v[1], R_32x4_3_1);
		X.v[1] ^= X.v[2];
	} if (Nrounds > 19) {
		X.v[0] += ks[0];
		X.v[1] += ks[1];
		X.v[2] += ks[2];
		X.v[3] += ks[3];
		X.v[4 - 1] += 5;
	} if (Nrounds > 20) {
		X.v[0] += X.v[1];
		X.v[1] = RotL_32(X.v[1], R_32x4_4_0);
		X.v[1] ^= X.v[0];
		X.v[2] += X.v[3];
		X.v[3] = RotL_32(X.v[3], R_32x4_4_1);
		X.v[3] ^= X.v[2];
	} if (Nrounds > 21) {
		X.v[0] += X.v[3];
		X.v[3] = RotL_32(X.v[3], R_32x4_5_0);
		X.v[3] ^= X.v[0];
		X.v[2] += X.v[1];
		X.v[1] = RotL_32(X.v[1], R_32x4_5_1);
		X.v[1] ^= X.v[2];
	} if (Nrounds > 22) {
		X.v[0] += X.v[1];
		X.v[1] = RotL_32(X.v[1], R_32x4_6_0);
		X.v[1] ^= X.v[0];
		X.v[2] += X.v[3];
		X.v[3] = RotL_32(X.v[3], R_32x4_6_1);
		X.v[3] ^= X.v[2];
	} if (Nrounds > 23) {
		X.v[0] += X.v[3];
		X.v[3] = RotL_32(X.v[3], R_32x4_7_0);
		X.v[3] ^= X.v[0];
		X.v[2] += X.v[1];
		X.v[1] = RotL_32(X.v[1], R_32x4_7_1);
		X.v[1] ^= X.v[2];
	} if (Nrounds > 23) {
		X.v[0] += ks[1];
		X.v[1] += ks[2];
		X.v[2] += ks[3];
		X.v[3] += ks[4];
		X.v[4 - 1] += 6;
	} if (Nrounds > 24) {
		X.v[0] += X.v[1];
		X.v[1] = RotL_32(X.v[1], R_32x4_0_0);
		X.v[1] ^= X.v[0];
		X.v[2] += X.v[3];
		X.v[3] = RotL_32(X.v[3], R_32x4_0_1);
		X.v[3] ^= X.v[2];
	} if (Nrounds > 25) {
		X.v[0] += X.v[3];
		X.v[3] = RotL_32(X.v[3], R_32x4_1_0);
		X.v[3] ^= X.v[0];
		X.v[2] += X.v[1];
		X.v[1] = RotL_32(X.v[1], R_32x4_1_1);
		X.v[1] ^= X.v[2];
	} if (Nrounds > 26) {
		X.v[0] += X.v[1];
		X.v[1] = RotL_32(X.v[1], R_32x4_2_0);
		X.v[1] ^= X.v[0];
		X.v[2] += X.v[3];
		X.v[3] = RotL_32(X.v[3], R_32x4_2_1);
		X.v[3] ^= X.v[2];
	} if (Nrounds > 27) {
		X.v[0] += X.v[3];
		X.v[3] = RotL_32(X.v[3], R_32x4_3_0);
		X.v[3] ^= X.v[0];
		X.v[2] += X.v[1];
		X.v[1] = RotL_32(X.v[1], R_32x4_3_1);
		X.v[1] ^= X.v[2];
	} if (Nrounds > 27) {
		X.v[0] += ks[2];
		X.v[1] += ks[3];
		X.v[2] += ks[4];
		X.v[3] += ks[0];
		X.v[4 - 1] += 7;
	} if (Nrounds > 28) {
		X.v[0] += X.v[1];
		X.v[1] = RotL_32(X.v[1], R_32x4_4_0);
		X.v[1] ^= X.v[0];
		X.v[2] += X.v[3];
		X.v[3] = RotL_32(X.v[3], R_32x4_4_1);
		X.v[3] ^= X.v[2];
	} if (Nrounds > 29) {
		X.v[0] += X.v[3];
		X.v[3] = RotL_32(X.v[3], R_32x4_5_0);
		X.v[3] ^= X.v[0];
		X.v[2] += X.v[1];
		X.v[1] = RotL_32(X.v[1], R_32x4_5_1);
		X.v[1] ^= X.v[2];
	} if (Nrounds > 30) {
		X.v[0] += X.v[1];
		X.v[1] = RotL_32(X.v[1], R_32x4_6_0);
		X.v[1] ^= X.v[0];
		X.v[2] += X.v[3];
		X.v[3] = RotL_32(X.v[3], R_32x4_6_1);
		X.v[3] ^= X.v[2];
	} if (Nrounds > 31) {
		X.v[0] += X.v[3];
		X.v[3] = RotL_32(X.v[3], R_32x4_7_0);
		X.v[3] ^= X.v[0];
		X.v[2] += X.v[1];
		X.v[1] = RotL_32(X.v[1], R_32x4_7_1);
		X.v[1] ^= X.v[2];
	} if (Nrounds > 31) {
		X.v[0] += ks[3];
		X.v[1] += ks[4];
		X.v[2] += ks[0];
		X.v[3] += ks[1];
		X.v[4 - 1] += 8;
	} if (Nrounds > 32) {
		X.v[0] += X.v[1];
		X.v[1] = RotL_32(X.v[1], R_32x4_0_0);
		X.v[1] ^= X.v[0];
		X.v[2] += X.v[3];
		X.v[3] = RotL_32(X.v[3], R_32x4_0_1);
		X.v[3] ^= X.v[2];
	} if (Nrounds > 33) {
		X.v[0] += X.v[3];
		X.v[3] = RotL_32(X.v[3], R_32x4_1_0);
		X.v[3] ^= X.v[0];
		X.v[2] += X.v[1];
		X.v[1] = RotL_32(X.v[1], R_32x4_1_1);
		X.v[1] ^= X.v[2];
	} if (Nrounds > 34) {
		X.v[0] += X.v[1];
		X.v[1] = RotL_32(X.v[1], R_32x4_2_0);
		X.v[1] ^= X.v[0];
		X.v[2] += X.v[3];
		X.v[3] = RotL_32(X.v[3], R_32x4_2_1);
		X.v[3] ^= X.v[2];
	} if (Nrounds > 35) {
		X.v[0] += X.v[3];
		X.v[3] = RotL_32(X.v[3], R_32x4_3_0);
		X.v[3] ^= X.v[0];
		X.v[2] += X.v[1];
		X.v[1] = RotL_32(X.v[1], R_32x4_3_1);
		X.v[1] ^= X.v[2];
	} if (Nrounds > 35) {
		X.v[0] += ks[4];
		X.v[1] += ks[0];
		X.v[2] += ks[1];
		X.v[3] += ks[2];
		X.v[4 - 1] += 9;
	} if (Nrounds > 36) {
		X.v[0] += X.v[1];
		X.v[1] = RotL_32(X.v[1], R_32x4_4_0);
		X.v[1] ^= X.v[0];
		X.v[2] += X.v[3];
		X.v[3] = RotL_32(X.v[3], R_32x4_4_1);
		X.v[3] ^= X.v[2];
	} if (Nrounds > 37) {
		X.v[0] += X.v[3];
		X.v[3] = RotL_32(X.v[3], R_32x4_5_0);
		X.v[3] ^= X.v[0];
		X.v[2] += X.v[1];
		X.v[1] = RotL_32(X.v[1], R_32x4_5_1);
		X.v[1] ^= X.v[2];
	} if (Nrounds > 38) {
		X.v[0] += X.v[1];
		X.v[1] = RotL_32(X.v[1], R_32x4_6_0);
		X.v[1] ^= X.v[0];
		X.v[2] += X.v[3];
		X.v[3] = RotL_32(X.v[3], R_32x4_6_1);
		X.v[3] ^= X.v[2];
	} if (Nrounds > 39) {
		X.v[0] += X.v[3];
		X.v[3] = RotL_32(X.v[3], R_32x4_7_0);
		X.v[3] ^= X.v[0];
		X.v[2] += X.v[1];
		X.v[1] = RotL_32(X.v[1], R_32x4_7_1);
		X.v[1] ^= X.v[2];
	} if (Nrounds > 39) {
		X.v[0] += ks[0];
		X.v[1] += ks[1];
		X.v[2] += ks[2];
		X.v[3] += ks[3];
		X.v[4 - 1] += 10;
	} if (Nrounds > 40) {
		X.v[0] += X.v[1];
		X.v[1] = RotL_32(X.v[1], R_32x4_0_0);
		X.v[1] ^= X.v[0];
		X.v[2] += X.v[3];
		X.v[3] = RotL_32(X.v[3], R_32x4_0_1);
		X.v[3] ^= X.v[2];
	} if (Nrounds > 41) {
		X.v[0] += X.v[3];
		X.v[3] = RotL_32(X.v[3], R_32x4_1_0);
		X.v[3] ^= X.v[0];
		X.v[2] += X.v[1];
		X.v[1] = RotL_32(X.v[1], R_32x4_1_1);
		X.v[1] ^= X.v[2];
	} if (Nrounds > 42) {
		X.v[0] += X.v[1];
		X.v[1] = RotL_32(X.v[1], R_32x4_2_0);
		X.v[1] ^= X.v[0];
		X.v[2] += X.v[3];
		X.v[3] = RotL_32(X.v[3], R_32x4_2_1);
		X.v[3] ^= X.v[2];
	} if (Nrounds > 43) {
		X.v[0] += X.v[3];
		X.v[3] = RotL_32(X.v[3], R_32x4_3_0);
		X.v[3] ^= X.v[0];
		X.v[2] += X.v[1];
		X.v[1] = RotL_32(X.v[1], R_32x4_3_1);
		X.v[1] ^= X.v[2];
	} if (Nrounds > 43) {
		X.v[0] += ks[1];
		X.v[1] += ks[2];
		X.v[2] += ks[3];
		X.v[3] += ks[4];
		X.v[4 - 1] += 11;
	} if (Nrounds > 44) {
		X.v[0] += X.v[1];
		X.v[1] = RotL_32(X.v[1], R_32x4_4_0);
		X.v[1] ^= X.v[0];
		X.v[2] += X.v[3];
		X.v[3] = RotL_32(X.v[3], R_32x4_4_1);
		X.v[3] ^= X.v[2];
	} if (Nrounds > 45) {
		X.v[0] += X.v[3];
		X.v[3] = RotL_32(X.v[3], R_32x4_5_0);
		X.v[3] ^= X.v[0];
		X.v[2] += X.v[1];
		X.v[1] = RotL_32(X.v[1], R_32x4_5_1);
		X.v[1] ^= X.v[2];
	} if (Nrounds > 46) {
		X.v[0] += X.v[1];
		X.v[1] = RotL_32(X.v[1], R_32x4_6_0);
		X.v[1] ^= X.v[0];
		X.v[2] += X.v[3];
		X.v[3] = RotL_32(X.v[3], R_32x4_6_1);
		X.v[3] ^= X.v[2];
	} if (Nrounds > 47) {
		X.v[0] += X.v[3];
		X.v[3] = RotL_32(X.v[3], R_32x4_7_0);
		X.v[3] ^= X.v[0];
		X.v[2] += X.v[1];
		X.v[1] = RotL_32(X.v[1], R_32x4_7_1);
		X.v[1] ^= X.v[2];
	} if (Nrounds > 47) {
		X.v[0] += ks[2];
		X.v[1] += ks[3];
		X.v[2] += ks[4];
		X.v[3] += ks[0];
		X.v[4 - 1] += 12;
	} if (Nrounds > 48) {
		X.v[0] += X.v[1];
		X.v[1] = RotL_32(X.v[1], R_32x4_0_0);
		X.v[1] ^= X.v[0];
		X.v[2] += X.v[3];
		X.v[3] = RotL_32(X.v[3], R_32x4_0_1);
		X.v[3] ^= X.v[2];
	} if (Nrounds > 49) {
		X.v[0] += X.v[3];
		X.v[3] = RotL_32(X.v[3], R_32x4_1_0);
		X.v[3] ^= X.v[0];
		X.v[2] += X.v[1];
		X.v[1] = RotL_32(X.v[1], R_32x4_1_1);
		X.v[1] ^= X.v[2];
	} if (Nrounds > 50) {
		X.v[0] += X.v[1];
		X.v[1] = RotL_32(X.v[1], R_32x4_2_0);
		X.v[1] ^= X.v[0];
		X.v[2] += X.v[3];
		X.v[3] = RotL_32(X.v[3], R_32x4_2_1);
		X.v[3] ^= X.v[2];
	} if (Nrounds > 51) {
		X.v[0] += X.v[3];
		X.v[3] = RotL_32(X.v[3], R_32x4_3_0);
		X.v[3] ^= X.v[0];
		X.v[2] += X.v[1];
		X.v[1] = RotL_32(X.v[1], R_32x4_3_1);
		X.v[1] ^= X.v[2];
	} if (Nrounds > 51) {
		X.v[0] += ks[3];
		X.v[1] += ks[4];
		X.v[2] += ks[0];
		X.v[3] += ks[1];
		X.v[4 - 1] += 13;
	} if (Nrounds > 52) {
		X.v[0] += X.v[1];
		X.v[1] = RotL_32(X.v[1], R_32x4_4_0);
		X.v[1] ^= X.v[0];
		X.v[2] += X.v[3];
		X.v[3] = RotL_32(X.v[3], R_32x4_4_1);
		X.v[3] ^= X.v[2];
	} if (Nrounds > 53) {
		X.v[0] += X.v[3];
		X.v[3] = RotL_32(X.v[3], R_32x4_5_0);
		X.v[3] ^= X.v[0];
		X.v[2] += X.v[1];
		X.v[1] = RotL_32(X.v[1], R_32x4_5_1);
		X.v[1] ^= X.v[2];
	} if (Nrounds > 54) {
		X.v[0] += X.v[1];
		X.v[1] = RotL_32(X.v[1], R_32x4_6_0);
		X.v[1] ^= X.v[0];
		X.v[2] += X.v[3];
		X.v[3] = RotL_32(X.v[3], R_32x4_6_1);
		X.v[3] ^= X.v[2];
	} if (Nrounds > 55) {
		X.v[0] += X.v[3];
		X.v[3] = RotL_32(X.v[3], R_32x4_7_0);
		X.v[3] ^= X.v[0];
		X.v[2] += X.v[1];
		X.v[1] = RotL_32(X.v[1], R_32x4_7_1);
		X.v[1] ^= X.v[2];
	} if (Nrounds > 55) {
		X.v[0] += ks[4];
		X.v[1] += ks[0];
		X.v[2] += ks[1];
		X.v[3] += ks[2];
		X.v[4 - 1] += 14;
	} if (Nrounds > 56) {
		X.v[0] += X.v[1];
		X.v[1] = RotL_32(X.v[1], R_32x4_0_0);
		X.v[1] ^= X.v[0];
		X.v[2] += X.v[3];
		X.v[3] = RotL_32(X.v[3], R_32x4_0_1);
		X.v[3] ^= X.v[2];
	} if (Nrounds > 57) {
		X.v[0] += X.v[3];
		X.v[3] = RotL_32(X.v[3], R_32x4_1_0);
		X.v[3] ^= X.v[0];
		X.v[2] += X.v[1];
		X.v[1] = RotL_32(X.v[1], R_32x4_1_1);
		X.v[1] ^= X.v[2];
	} if (Nrounds > 58) {
		X.v[0] += X.v[1];
		X.v[1] = RotL_32(X.v[1], R_32x4_2_0);
		X.v[1] ^= X.v[0];
		X.v[2] += X.v[3];
		X.v[3] = RotL_32(X.v[3], R_32x4_2_1);
		X.v[3] ^= X.v[2];
	} if (Nrounds > 59) {
		X.v[0] += X.v[3];
		X.v[3] = RotL_32(X.v[3], R_32x4_3_0);
		X.v[3] ^= X.v[0];
		X.v[2] += X.v[1];
		X.v[1] = RotL_32(X.v[1], R_32x4_3_1);
		X.v[1] ^= X.v[2];
	} if (Nrounds > 59) {
		X.v[0] += ks[0];
		X.v[1] += ks[1];
		X.v[2] += ks[2];
		X.v[3] += ks[3];
		X.v[4 - 1] += 15;
	} if (Nrounds > 60) {
		X.v[0] += X.v[1];
		X.v[1] = RotL_32(X.v[1], R_32x4_4_0);
		X.v[1] ^= X.v[0];
		X.v[2] += X.v[3];
		X.v[3] = RotL_32(X.v[3], R_32x4_4_1);
		X.v[3] ^= X.v[2];
	} if (Nrounds > 61) {
		X.v[0] += X.v[3];
		X.v[3] = RotL_32(X.v[3], R_32x4_5_0);
		X.v[3] ^= X.v[0];
		X.v[2] += X.v[1];
		X.v[1] = RotL_32(X.v[1], R_32x4_5_1);
		X.v[1] ^= X.v[2];
	} if (Nrounds > 62) {
		X.v[0] += X.v[1];
		X.v[1] = RotL_32(X.v[1], R_32x4_6_0);
		X.v[1] ^= X.v[0];
		X.v[2] += X.v[3];
		X.v[3] = RotL_32(X.v[3], R_32x4_6_1);
		X.v[3] ^= X.v[2];
	} if (Nrounds > 63) {
		X.v[0] += X.v[3];
		X.v[3] = RotL_32(X.v[3], R_32x4_7_0);
		X.v[3] ^= X.v[0];
		X.v[2] += X.v[1];
		X.v[1] = RotL_32(X.v[1], R_32x4_7_1);
		X.v[1] ^= X.v[2];
	} if (Nrounds > 63) {
		X.v[0] += ks[1];
		X.v[1] += ks[2];
		X.v[2] += ks[3];
		X.v[3] += ks[4];
		X.v[4 - 1] += 16;
	} if (Nrounds > 64) {
		X.v[0] += X.v[1];
		X.v[1] = RotL_32(X.v[1], R_32x4_0_0);
		X.v[1] ^= X.v[0];
		X.v[2] += X.v[3];
		X.v[3] = RotL_32(X.v[3], R_32x4_0_1);
		X.v[3] ^= X.v[2];
	} if (Nrounds > 65) {
		X.v[0] += X.v[3];
		X.v[3] = RotL_32(X.v[3], R_32x4_1_0);
		X.v[3] ^= X.v[0];
		X.v[2] += X.v[1];
		X.v[1] = RotL_32(X.v[1], R_32x4_1_1);
		X.v[1] ^= X.v[2];
	} if (Nrounds > 66) {
		X.v[0] += X.v[1];
		X.v[1] = RotL_32(X.v[1], R_32x4_2_0);
		X.v[1] ^= X.v[0];
		X.v[2] += X.v[3];
		X.v[3] = RotL_32(X.v[3], R_32x4_2_1);
		X.v[3] ^= X.v[2];
	} if (Nrounds > 67) {
		X.v[0] += X.v[3];
		X.v[3] = RotL_32(X.v[3], R_32x4_3_0);
		X.v[3] ^= X.v[0];
		X.v[2] += X.v[1];
		X.v[1] = RotL_32(X.v[1], R_32x4_3_1);
		X.v[1] ^= X.v[2];
	} if (Nrounds > 67) {
		X.v[0] += ks[2];
		X.v[1] += ks[3];
		X.v[2] += ks[4];
		X.v[3] += ks[0];
		X.v[4 - 1] += 17;
	} if (Nrounds > 68) {
		X.v[0] += X.v[1];
		X.v[1] = RotL_32(X.v[1], R_32x4_4_0);
		X.v[1] ^= X.v[0];
		X.v[2] += X.v[3];
		X.v[3] = RotL_32(X.v[3], R_32x4_4_1);
		X.v[3] ^= X.v[2];
	} if (Nrounds > 69) {
		X.v[0] += X.v[3];
		X.v[3] = RotL_32(X.v[3], R_32x4_5_0);
		X.v[3] ^= X.v[0];
		X.v[2] += X.v[1];
		X.v[1] = RotL_32(X.v[1], R_32x4_5_1);
		X.v[1] ^= X.v[2];
	} if (Nrounds > 70) {
		X.v[0] += X.v[1];
		X.v[1] = RotL_32(X.v[1], R_32x4_6_0);
		X.v[1] ^= X.v[0];
		X.v[2] += X.v[3];
		X.v[3] = RotL_32(X.v[3], R_32x4_6_1);
		X.v[3] ^= X.v[2];
	} if (Nrounds > 71) {
		X.v[0] += X.v[3];
		X.v[3] = RotL_32(X.v[3], R_32x4_7_0);
		X.v[3] ^= X.v[0];
		X.v[2] += X.v[1];
		X.v[1] = RotL_32(X.v[1], R_32x4_7_1);
		X.v[1] ^= X.v[2];
	} if (Nrounds > 71) {
		X.v[0] += ks[3];
		X.v[1] += ks[4];
		X.v[2] += ks[0];
		X.v[3] += ks[1];
		X.v[4 - 1] += 18;
	} 
	return X;
} 

template <class T>
__kernel void PRNG_threefry4x32(
        __global uint4 *randomnumber,
        threefry4x32_ctr_t ctr_i,
        T inf,
        T sup,
        T threshold,
        uint nrounds,
        uint numrandom
){
        size_t  gdx = get_global_id(0);

        uint maxUint = 0;
        maxUint--;
        float r = (float)maxUint;

        threefry4x32_ctr_t      ctr = ctr_i; 
        threefry4x32_ukey_t ukey;

        ukey.v[0] = ukey.v[1] = ukey.v[2] = ukey.v[3] = gdx;

        threefry4x32_ctr_t  random4;

        if ( gdx < numrandom )
        {
                random4 = threefry4x32_R(nrounds, ctr, ukey);
                uint4 frnd;
				
                frnd.x = ( (((float)random4.v[0]) / r) * (sup - inf) + inf ) > threshold? 1 : 0;
                frnd.y = ( (((float)random4.v[1]) / r) * (sup - inf) + inf ) > threshold? 1 : 0;
                frnd.z = ( (((float)random4.v[2]) / r) * (sup - inf) + inf ) > threshold? 1 : 0;
                frnd.w = ( (((float)random4.v[3]) / r) * (sup - inf) + inf ) > threshold? 1 : 0;
				
                randomnumber[gdx] = frnd;
        }
}


template __attribute__((mangled_name(RNGBernoulli_float))) __kernel void PRNG_threefry4x32(__global uint4 *randomnumber, threefry4x32_ctr_t ctr_i, float inf, float sup, float threshold, uint nrounds, uint numrandonm);

template __attribute__((mangled_name(RNGBernoulli_double))) __kernel void PRNG_threefry4x32(__global uint4 *randomnumber, threefry4x32_ctr_t ctr_i, double inf, double sup, double threshold, uint nrounds, uint numrandonm);

//end of the looooooong gpu_random_generator kernel 


template <class T>
__kernel void OCL_memset(__global T* buffer, const T value, const int size){
	int gdx = get_global_id(0);
	if(gdx < size){
		buffer[gdx] = value;	
	}
}

template __attribute__((mangled_name(oclmem_float))) __kernel void OCL_memset(__global float* buffer, const float value, const int size);
template __attribute__((mangled_name(oclmem_double))) __kernel void OCL_memset(__global double* buffer, const double value, const int size);

__kernel void OCL_memset2(__global int* buffer, const int value, const int size){
        int gdx = get_global_id(0);
        if(gdx < size){
                buffer[gdx] = value;    
        }
}

template <class T>
__kernel void caffe_gpu_sign(const int N, __global T* X, __global T* Y){
     int gdx = get_global_id(0);
     if(gdx < N){
          Y[gdx] =((0.0<X[gdx])-(X[gdx]<0.0));
     }
}

template __attribute__((mangled_name(caffe_gpu_sign_float))) __kernel void caffe_gpu_sign(const int N, __global float* X, __global float* Y);
template __attribute__((mangled_name(caffe_gpu_sign_double))) __kernel void caffe_gpu_sign(const int N, __global double* X, __global double* Y);


template <class T>
__kernel void get_max(const int num, const int dim, __global T* data, __global T* out){
     int index = get_global_id(0);
     if (index < num) {
	T maxval = -FLT_MAX;
        for (int i = 0; i <  dim; i++)
	maxval = max( data[index*dim + i], maxval );
        out[index] = maxval;
      }
}

template __attribute__ ((mangled_name(get_max_float))) __kernel void get_max(const int num, const int dim, __global float* data, __global float* out);
template __attribute__ ((mangled_name(get_max_double))) __kernel void get_max(const int num, const int dim, __global double* data, __global double* out);

template <class T>
__kernel void exp (const int num, __global T* data, __global T* out){
        int index = get_global_id(0);
        if (index < num) 
        out[index] = exp(data[index]);
}

template __attribute__ ((mangled_name(exp_float))) __kernel void exp (const int num, __global float* data, __global float* out);
template __attribute__ ((mangled_name(exp_double))) __kernel void exp (const int num, __global double* data, __global double* out);



template <class T>
__kernel void kernel_channel_max(const int num, const int channels,
    const int spatial_dim, __global const T* data, __global T* out) {
    int index = get_global_id(0);
    if(index < num * spatial_dim) {
    int n = index / spatial_dim;
    int s = index % spatial_dim;
    T maxval = -FLT_MAX;
    for (int c = 0; c < channels; ++c) {
      maxval = max(data[(n * channels + c) * spatial_dim + s], maxval);
    }
    out[index] = maxval;
  }
}

template __attribute__ ((mangled_name(kernel_channel_max_float))) __kernel void kernel_channel_max(const int num, const int channels,
    const int spatial_dim, __global const float* data, __global float* out);
template __attribute__ ((mangled_name(kernel_channel_max_double))) __kernel void kernel_channel_max(const int num, const int channels,
    const int spatial_dim, __global const double* data, __global  double* out);

template <class T>
__kernel void kernel_channel_subtract(const int count,
    const int num, const int channels,
    const int spatial_dim, __global const T* channel_max, __global T* data) {
    int index = get_global_id(0);
    if(index < count) {
    int n = index / channels / spatial_dim;
    int s = index % spatial_dim;
    data[index] -= channel_max[n * spatial_dim + s];
  }
}

template __attribute__ ((mangled_name(kernel_channel_subtract_float))) __kernel void kernel_channel_subtract(const int count, const int num, const int channels, const int spatial_dim, __global const float* channel_max, __global float* data);
template __attribute__ ((mangled_name(kernel_channel_subtract_double))) __kernel void kernel_channel_subtract(const int count, const int num, const int channels, const int spatial_dim, __global const double* channel_max, __global double* data);

template <class T>
__kernel void kernel_exp(const int count, __global const T* data, __global T* out) {
 int index = get_global_id(0);
   if(index < count) {
    out[index] = exp(data[index]);
  }
}

template __attribute__ ((mangled_name(kernel_exp_float))) __kernel void kernel_exp(const int count, __global const float* data, __global float* out);
template __attribute__ ((mangled_name(kernel_exp_double))) __kernel void kernel_exp(const int count, __global const double* data, __global double* out);

template <class T>
__kernel void kernel_channel_sum(const int num, const int channels,
    const int spatial_dim, __global const T* data, __global T* channel_sum) {
  int index = get_global_id(0);
   if(index < num * spatial_dim) {
    int n = index / spatial_dim;
    int s = index % spatial_dim;
    T sum = 0;
    for (int c = 0; c < channels; ++c) {
      sum += data[(n * channels + c) * spatial_dim + s];
    }
    channel_sum[index] = sum;
  }
}

template __attribute__ ((mangled_name(kernel_channel_sum_float))) __kernel void kernel_channel_sum(const int num, const int channels,
    const int spatial_dim, __global const float* data, __global float* channel_sum);
template __attribute__ ((mangled_name(kernel_channel_sum_double))) __kernel void kernel_channel_sum(const int num, const int channels,
    const int spatial_dim, __global const double* data, __global double* channel_sum);

template <class T>
__kernel void kernel_channel_div(const int count,
    const int num, const int channels,
    const int spatial_dim, __global const T* channel_sum, __global T* data) {
    int index = get_global_id(0);
   if(index < count) {
    int n = index / channels / spatial_dim;
    int s = index % spatial_dim;
    data[index] /= channel_sum[n * spatial_dim + s];
  }
}

template __attribute__ ((mangled_name(kernel_channel_div_float))) __kernel void kernel_channel_div(const int count,
    const int num, const int channels,
    const int spatial_dim, __global const float* channel_sum, __global float* data);
template __attribute__ ((mangled_name(kernel_channel_div_double))) __kernel void kernel_channel_div(const int count,
    const int num, const int channels,
    const int spatial_dim, __global const double* channel_sum, __global double* data);

template <class T>
__kernel void kernel_channel_dot(const int num, const int channels,
    const int spatial_dim, __global const T* data_1, __global const T* data_2,
    __global T* channel_dot) {
    int index = get_global_id(0);
    if(index < num * spatial_dim) {
        int n = index / spatial_dim;
        int s = index % spatial_dim;
        T dot = 0;
        for (int c = 0; c < channels; ++c) {
            dot += (data_1[(n * channels + c) * spatial_dim + s]
                 * data_2[(n * channels + c) * spatial_dim + s]);
        }
        channel_dot[index] = dot;
    }
}

template __attribute__ ((mangled_name(kernel_channel_dot_float))) __kernel void kernel_channel_dot(const int num, const int channels,
    const int spatial_dim, __global const float* data_1, __global const float* data_2,
    __global float* channel_dot);
template __attribute__ ((mangled_name(kernel_channel_dot_double))) __kernel void kernel_channel_dot(const int num, const int channels,
    const int spatial_dim, __global const double* data_1, __global const double* data_2,
    __global double* channel_dot);



template <class T>
__kernel void diff (const int num, const int dim, __global T* data, __global T* label){
        int index = get_global_id(0);
        int total = get_global_size(0);
        int offset;
	for(index; index < num; index +=  total){
  	offset = (int) label[index];
        data[index * dim + offset] -= 1;
        }
}

template __attribute__ ((mangled_name(diff_float))) __kernel void diff (const int num, const int dim, __global float* data, __global float* label);
template __attribute__ ((mangled_name(diff_double))) __kernel void diff (const int num, const int dim, __global double* data, __global double* label);

template <class T>
__kernel void scal (const int num, const T alpha, __global T* data){
        int index = get_global_id(0);
        int total = get_global_size(0);
        for(index; index < num; index +=  total){
        data[index] = data[index] * alpha;
        }
}

template __attribute__ ((mangled_name(scal_float))) __kernel void scal (const int num, const float alpha,  __global float* data);
template __attribute__ ((mangled_name(scal_double))) __kernel void scal (const int num, const double alpha,  __global double* data);

template <class T>
__kernel void div (const int n, __global const T* a, __global const T* b, __global T* y){
	int index = get_global_id(0);
        if (index < n)
        y[index] = a[index] / b[index];
}

template __attribute__ ((mangled_name(div_float))) __kernel void div (const int n, __global const float* a, __global const float* b, __global float* y);
//template __attribute__ ((mangled_name(div_double))) __kernel void div (const int n, __global const double* a, __global const double* b, __global double* y);

template <class T>
__kernel void add_scalar (const int n, const T alpha, __global T* y){
        int index = get_global_id(0);
        if (index < n)
        y[index] += alpha;
}

template __attribute__ ((mangled_name(add_scalar_float))) __kernel void add_scalar (const int n, const float alpha, __global float* y);
template __attribute__ ((mangled_name(add_scalar_double))) __kernel void add_scalar (const int n, const double alpha, __global double* y);

template <typename Dtype>
__kernel void caffe_gpu_add(const int n, const Dtype* in1, const Dtype* in2, Dtype* y){
        int index = get_global_id(0);
        if (index < n)
        y[index] = in1[index] + in2[index] ;
}
template __attribute__ ((mangled_name(caffe_gpu_add_float))) __kernel void caffe_gpu_add(const int n, const float* in1, const float* in2, float* y);
template __attribute__ ((mangled_name(caffe_gpu_add_double))) __kernel void caffe_gpu_add(const int n, const double* in1, const double* in2, double* y);

template <class T>
__kernel void element_mul (const int n, __global const T* a, __global const T* b, __global T* y){
        int index = get_global_id(0);
       if (index < n)
        y[index] = a[index] * b[index];
}

template __attribute__ ((mangled_name(element_mul_float))) __kernel void element_mul (const int n, __global const float* a, __global const float* b, __global float* y);
template __attribute__ ((mangled_name(element_mul_double))) __kernel void element_mul (const int n,__global const double* a, __global const double* b, __global double* y);


template <class T>
__kernel void powx (const int n, __global const T* a, const T alpha, __global T* y){
        int index = get_global_id(0);
        if (index < n)
//           y[index] = a[index] + alpha;
           y[index] = pow(a[index], alpha);
}

template __attribute__ ((mangled_name(powx_float))) __kernel void powx (const int n, __global const float* a, const float alpha, __global float* y); 
template __attribute__ ((mangled_name(powx_double))) __kernel void powx (const int n, __global const double* a, const double alpha, __global double* y); 


template <class T>
__kernel void transpose(__global const T *src, __global T* dst, int width, int height, int optnum){
     int gidx = get_global_id(0);
     int gidy = get_global_id(1);
     int gidyy = gidy;
     int index = gidy / height;
     int offset = index * width * height;
     gidy = gidy % height;
     if( gidx < width && gidyy < height * optnum )
         dst[offset + height * gidx + gidy] = src[offset + width * gidy + gidx];
}
template __attribute__((mangled_name(transposefloat))) __kernel void transpose(__global const float* src, __global float* dst, const int width, const int height, int optnum); 
template __attribute__((mangled_name(transposedouble))) __kernel void transpose(__global const double* src, __global double* dst, const int width, const int heighti, int optnum);

template <class T>
__kernel void transform(__global const T *src, __global T* dst, int top_offset, int width, int height, int optnum){
     int gidx = get_global_id(0);
     int index;
     index = (optnum==1) ? 0: gidx % optnum;
     dst = dst + top_offset; // now we point at (*top)[n]
     int offset = gidx / optnum;
     int i = 0;
     for(i = 0 ; i < width; i++)
         dst[(index * height + offset)* width + i] = src[gidx * width + i];
}
template __attribute__((mangled_name(transformfloat))) __kernel void transform(__global const float* src, __global float* dst, int top_offset, const int width, const int height, const int optnum); 
template __attribute__((mangled_name(transformdouble))) __kernel void transform(__global const double* src, __global double* dst, int top_offset, const int width, const int height, const int optnum); 
