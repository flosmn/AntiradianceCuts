#include "bvh.h"
#include "Utils/stream.h"

#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/scan.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>

#include <limits>

#include "CudaResources/cudaUtil.hpp"
#include "CudaResources/cudaTimer.hpp"

template<typename T>
struct Max {
	__host__ __device__ T operator()(T const& a, T const& b) const { return fmaxf(a, b) ;}
};

template<typename T>
struct Min { 
	__host__ __device__ T operator()(T const& a, T const& b) const { return fminf(a, b) ;}
};

struct Normalize {
	Normalize(float3 const& min, float3 const& max) : m_min(min), m_max(max) {}

	__host__ __device__ float3 operator() (float3 const& v) {
		return (v - m_min) / (m_max - m_min);
	}

	float3 m_min, m_max;
};

// Expands a 10-bit integer into 30 bits
// by inserting 2 zeros after each bit.
__inline__ __host__ __device__
unsigned int expandBits(unsigned int v)
{
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

struct MortonCode {
	__host__ __device__ unsigned int operator() (float3 const& v) {
		// Calculates a 30-bit Morton code for the
		// given 3D point located within the unit cube [0,1].
	    float x = fminf(fmaxf(v.x * 1024.0f, 0.0f), 1023.0f);
	    float y = fminf(fmaxf(v.y * 1024.0f, 0.0f), 1023.0f);
	    float z = fminf(fmaxf(v.z * 1024.0f, 0.0f), 1023.0f);
	    unsigned int xx = expandBits((unsigned int)x);
	    unsigned int yy = expandBits((unsigned int)y);
	    unsigned int zz = expandBits((unsigned int)z);
	    return xx * 4 + yy * 2 + zz;
	}
};

inline __device__ int delta(int i, int j, int* morton, int N)
{
	if (j < 0 || j >= N) {
		return -1;
	}
	const int d = __clz(morton[i] ^ morton[j]); 
	if (d==0) return __clz(i ^ j);
	return d;
}

inline __device__ int sign(int i)
{
	return (i < 0) ? -1 : 1; 
}

// TODO: shared memory
__global__ void kernel_buildRadixTree(int* morton, Node* nodes, int* parents, int N)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= N) {
		return;
	}
	int t = 0; // while-loop variable

	// determine direction of range
	const int d = sign(delta(i, i+1, morton, N) - delta(i, i-1, morton, N));

	// compute upper bound for the lenght of the range	
	const int delta_min = delta(i, i - d, morton, N);
	int l_max = 128;
	while (delta(i, i+l_max*d, morton, N) > delta_min) {
		l_max *= 4;
	}

	// find the other end with binary search
	int l = 0;
	t = l_max/2;
	do {
		if (delta(i, i+(l+t)*d, morton, N) > delta_min) {
			l += t;
		}
		t /= 2;
	} while (t > 0);
	const int j = i + l*d;
	
	// find split position with binary search
	const int delta_node = delta(i, j, morton, N);
	int s = 0;
	int k = 2;
	t = (l+k-1)/k;
	do {
		if (delta(i, i + (s+t)*d, morton, N) > delta_node) {
			s += t;
		}
		k *= 2;
		t = (l+k-1)/k;
	} while (t > 0);
	const int split = i + s*d + min(d, 0);

	// output child pointers
	Node node;
	node.left	= min(i, j) == split	? split		: -split;
	node.right	= max(i, j) == split+1	? (split+1)	: -(split+1);

	if (node.left >= 0) {
		parents[node.left] = i;
	} else {
		parents[N - node.left - 1] = i;
	}
	if (node.right >= 0) {
		parents[node.right] = i;
	} else {
		parents[N - node.right - 1] = i;
	}

	nodes[i] = node;
}

// TODO: shared memory
__global__ void kernel_boundingBoxes(Node* nodes, int* parents, int* ids, float3* positions, int N)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= N) {
		return;
	}

	int parent = parents[i];	

	bool finished = false;
	while (!finished) {
		Node& node = nodes[parent];
		if (atomicAdd(&node.visited, 1) == 0) {
			return;
		} else {
			const int left = node.left;
			const int right = node.right;
			const float3 bbMinLeft	= left < 0	? nodes[-left].bbMin	: positions[ids[left]];
			const float3 bbMaxLeft	= left < 0	? nodes[-left].bbMax	: positions[ids[left]];
			const float3 bbMinRight = right < 0 ? nodes[-right].bbMin	: positions[ids[right]];
			const float3 bbMaxRight = right < 0 ? nodes[-right].bbMax	: positions[ids[right]];
			node.bbMin = fminf(bbMinLeft, bbMinRight); 
			node.bbMax = fmaxf(bbMaxLeft, bbMaxRight); 

			if (parent == 0) {
				finished = true;
			}
		}

		parent = parents[N - 1 + parent];
	}
}

__global__ void kernel_traverse(Node* nodes, int* ids, float3* positions,
	float3* bbMinDebug, float3* bbMaxDebug, int N)
{
	int stack[32];
	stack[0] = 0;
	int stack_ptr = 1;

	while (stack_ptr > 0)
	{
		stack_ptr--;
		int index = stack[stack_ptr];
		Node node = nodes[index];
		
		bbMinDebug[N + index] = node.bbMin;
		bbMaxDebug[N + index] = node.bbMax;
		
		if (node.right < 0) {
			// inner node
			stack[stack_ptr] = -(node.right);
			stack_ptr++;
		} else { 
			// leaf
			bbMinDebug[node.right] = positions[ids[node.right]];
			bbMaxDebug[node.right] = positions[ids[node.right]];
		}
		if (node.left < 0) {
			// inner node
			stack[stack_ptr] = -(node.left);
			stack_ptr++;
		} else {
			// leaf
			bbMinDebug[node.left] = positions[ids[node.left]];
			bbMaxDebug[node.left] = positions[ids[node.left]];
		}
	}
}

BVH::BVH(std::vector<glm::vec3> const& positions)
{
	const int numElements = positions.size();
	m_data.positions.resize(numElements);
	m_data.morton.resize(numElements);
	m_data.ids.resize(numElements);
	m_data.parents.resize(2 * numElements - 1);

	for (int i = 0; i < positions.size(); ++i)
	{
		m_data.positions[i] = make_float3(positions[i]);
	}

	cuda::CudaTimer timer;
	timer.start();
	float3 min = thrust::reduce(m_data.positions.begin(), m_data.positions.end(), 
			make_float3(std::numeric_limits<float>::max()), Min<float3>());
	float3 max = thrust::reduce(m_data.positions.begin(), m_data.positions.end(),
			make_float3(std::numeric_limits<float>::min()), Max<float3>());
	
	thrust::counting_iterator<int> iter(0);
	thrust::copy(iter, iter + m_data.ids.size(), m_data.ids.begin());
	
	thrust::device_vector<float3> normalized(numElements);
	thrust::transform(m_data.positions.begin(), m_data.positions.end(), normalized.begin(), Normalize(min, max));
	thrust::transform(normalized.begin(), normalized.end(), m_data.morton.begin(), MortonCode());

	//std::cout << "morton codes and ids before sort: " << std::endl;
	//thrust::copy(m_data.morton.begin(), m_data.morton.end(), std::ostream_iterator<int>(std::cout, "\n"));
	//thrust::copy(m_data.ids.begin(), m_data.ids.end(), std::ostream_iterator<int>(std::cout, "\n"));

	cuda::CudaTimer timerSort;
	timerSort.start();
	thrust::sort_by_key(m_data.morton.begin(), m_data.morton.end(), m_data.ids.begin());
	timerSort.stop();

	//std::cout << "morton codes and ids after sort: " << std::endl;
	//thrust::copy(m_data.morton.begin(), m_data.morton.end(), std::ostream_iterator<int>(std::cout, "\n"));
	//thrust::copy(m_data.ids.begin(), m_data.ids.end(), std::ostream_iterator<int>(std::cout, "\n"));

	m_nodes.resize(numElements - 1);

	cuda::CudaTimer timerBuildRadixTree;
	timerBuildRadixTree.start();
	{
		dim3 dimBlock(128);
		dim3 dimGrid((numElements + dimBlock.x - 1) / dimBlock.x);
		kernel_buildRadixTree<<<dimGrid, dimBlock>>>(
			thrust::raw_pointer_cast(&m_data.morton[0]),
			thrust::raw_pointer_cast(&m_nodes[0]),
			thrust::raw_pointer_cast(&m_data.parents[0]),
			numElements);
	}
	timerBuildRadixTree.stop();

	cuda::CudaTimer timerBoundingBoxes;
	timerBoundingBoxes.start();
	{
		dim3 dimBlock(128);
		dim3 dimGrid((numElements + dimBlock.x - 1) / dimBlock.x);
		kernel_boundingBoxes<<<dimGrid, dimBlock>>>(
			thrust::raw_pointer_cast(&m_nodes[0]),
			thrust::raw_pointer_cast(&m_data.parents[0]),
			thrust::raw_pointer_cast(&m_data.ids[0]),
			thrust::raw_pointer_cast(&m_data.positions[0]),
			numElements);
	}
	timerBoundingBoxes.stop();

	//std::cout << "parents: " << std::endl;
	//thrust::copy(m_data.parents.begin(), m_data.parents.end(), std::ostream_iterator<int>(std::cout, "\n"));

	timer.stop();
	std::cout << "bvh creation: " << timer.getTime() << std::endl;
	std::cout << "sort: " << timerSort.getTime() << std::endl;
	std::cout << "kernel build radix-tree: " << timerBuildRadixTree.getTime() << std::endl;
	std::cout << "kernel build bounding boxes: " << timerBoundingBoxes.getTime() << std::endl;

	std::cout << "radix tree: " << std::endl;
	for (int i = 0; i < m_nodes.size(); ++i)
	{
		Node node = m_nodes[i];
		std::cout << "node " << i << ": left " << node.left << ", right " << node.right << std::endl;
	}
	
	thrust::device_vector<float3> bbMinDebug(2 * numElements - 1);
	thrust::device_vector<float3> bbMaxDebug(2 * numElements - 1);
	{
		int stack[32];
		stack[0] = 0;
		int stack_ptr = 1;

		while (stack_ptr > 0)
		{
			stack_ptr--;
			int index = stack[stack_ptr];
			Node node = m_nodes[index];
			
			bbMinDebug[numElements + index] = node.bbMin;
			bbMaxDebug[numElements + index] = node.bbMax;
			
			if (node.right < 0) {
				// inner node
				stack[stack_ptr] = -(node.right);
				stack_ptr++;
			} else { 
				// leaf
				bbMinDebug[node.right] = m_data.positions[m_data.ids[node.right]];
				bbMaxDebug[node.right] = m_data.positions[m_data.ids[node.right]];
			}
			if (node.left < 0) {
				// inner node
				stack[stack_ptr] = -(node.left);
				stack_ptr++;
			} else {
				// leaf
				bbMinDebug[node.left] = m_data.positions[m_data.ids[node.left]];
				bbMaxDebug[node.left] = m_data.positions[m_data.ids[node.left]];
			}
			
		}
	}

	std::cout << "bouning boxes: " << std::endl;
	for (int i = 0; i < bbMinDebug.size(); ++i) {
		std::cout << "min: " << make_vec3(bbMinDebug[i]) << ", max: " << make_vec3(bbMaxDebug[i]) << std::endl;
	}
}

BVH::~BVH()
{
}
