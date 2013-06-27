#ifndef MORTON_H_
#define MORTON_H_

#include <stdint.h>

#include <thrust/functional.h>

template<typename T>
void PrintBinary(T i)
{
	T one = 1;
	T shift = sizeof(T) * 8 - 1;
	for (T z = (one << shift); z > 0; z >>= 1)
	{
		if ((i & z) == z) {
			std::cout << "1";
		} else {
			std::cout << "0";
		}
	}
	std::cout << std::endl;
}

inline __host__ __device__ uint32_t expandBitsBy1(uint32_t x)
{
	x &= 0x0000ffff;                  // x = ---- ---- ---- ---- fedc ba98 7654 3210
	x = (x ^ (x <<  8)) & 0x00ff00ff; // x = ---- ---- fedc ba98 ---- ---- 7654 3210
	x = (x ^ (x <<  4)) & 0x0f0f0f0f; // x = ---- fedc ---- ba98 ---- 7654 ---- 3210
	x = (x ^ (x <<  2)) & 0x33333333; // x = --fe --dc --ba --98 --76 --54 --32 --10
	x = (x ^ (x <<  1)) & 0x55555555; // x = -f-e -d-c -b-a -9-8 -7-6 -5-4 -3-2 -1-0
	return x;
}

inline __host__ __device__ uint32_t expandBitsBy2(uint32_t x)
{
	x &= 0x000003ff;                  // x = ---- ---- ---- ---- ---- --98 7654 3210
	x = (x ^ (x << 16)) & 0xff0000ff; // x = ---- --98 ---- ---- ---- ---- 7654 3210
	x = (x ^ (x <<  8)) & 0x0300f00f; // x = ---- --98 ---- ---- 7654 ---- ---- 3210
	x = (x ^ (x <<  4)) & 0xc30c30c3; // x = ---- --98 ---- 76-- --54 ---- 32-- --10
	x = (x ^ (x <<  2)) & 0x49249249; // x = ---- 9--8 --7- -6-- 5--4 --3- -2-- 1--0
	return x;
}

inline __host__ __device__ uint32_t interleave(uint32_t x, uint32_t y) {
	return (expandBitsBy1(y) << 1) + expandBitsBy1(x);
}

inline __host__ __device__ uint32_t interleave(uint32_t x, uint32_t y, uint32_t z) {
	return (expandBitsBy2(z) << 2) + (expandBitsBy2(y) << 1) + expandBitsBy2(x);
}

inline __host__ __device__ uint32_t interleave(uint32_t x, uint32_t y, uint32_t z, uint32_t u, uint32_t v, uint32_t w) {
	uint32_t res = 0;

	for (int i = 0, k = 0; i < 5; ++i, k = k+5) // unroll for more speed...
	{
	  res |= (x & 1U << i) << k
		  |  (y & 1U << i) << (k + 1)
		  |  (z & 1U << i) << (k + 2)
		  |  (u & 1U << i) << (k + 3)
		  |  (v & 1U << i) << (k + 4)
		  |  (w & 1U << i) << (k + 5);
	}
	return res;
}

inline __host__ __device__ uint64_t interleave(uint64_t x, uint64_t y, uint64_t z, uint64_t u, uint64_t v, uint64_t w) {
	uint64_t res = 0;
	uint64_t one = 1U;
	for (int i = 0, k = 0; i < 10; ++i, k = k+5) // unroll for more speed...
	{
	  res |= (x & one << i) << k
		  |  (y & one << i) << (k + 1)
		  |  (z & one << i) << (k + 2)
		  |  (u & one << i) << (k + 3)
		  |  (v & one << i) << (k + 4)
		  |  (w & one << i) << (k + 5);
	}
	return res;
}

struct MortonCode3D {
	__host__ __device__ uint32_t operator() (float3 const& v) {
		// Calculates a 30-bit Morton code for the
		// given 3D point located within the unit cube [0,1].
	    float x = fminf(fmaxf(v.x * 1024.0f, 0.0f), 1023.0f);
	    float y = fminf(fmaxf(v.y * 1024.0f, 0.0f), 1023.0f);
	    float z = fminf(fmaxf(v.z * 1024.0f, 0.0f), 1023.0f);
	    return interleave(uint32_t(x), uint32_t(y), uint32_t(z));
	}
	
};

struct MortonCode6D : public thrust::binary_function<uint64_t, float3, float3>
{
	__host__ __device__ uint64_t operator() (float3 const& a, float3 const& b) {
		// Calculates a 30-bit Morton code for the
		// given 3D point located within the unit cube [0,1].
	    float x = fminf(fmaxf(a.x * 1024.0f, 0.0f), 1023.0f);
	    float y = fminf(fmaxf(a.y * 1024.0f, 0.0f), 1023.0f);
	    float z = fminf(fmaxf(a.z * 1024.0f, 0.0f), 1023.0f);
	    float u = fminf(fmaxf(b.x * 1024.0f, 0.0f), 1023.0f);
	    float v = fminf(fmaxf(b.y * 1024.0f, 0.0f), 1023.0f);
	    float w = fminf(fmaxf(b.z * 1024.0f, 0.0f), 1023.0f);
	    return interleave(uint64_t(x), uint64_t(y), uint64_t(z), uint64_t(u), uint64_t(v), uint64_t(w));
	}
};

#endif // MORTON_H_
