#include "bvh.h"
#include "Utils/stream.h"
#include "morton.h"

#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/scan.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>
#include <thrust/host_vector.h>

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

inline __device__ int delta(int i, int j, uint64_t* morton, int numNodes)
{
	if (j < 0 || j > numNodes) {
		return -1;
	}
	const int d = __clzll(morton[i] ^ morton[j]); 
	if (d==0) return __clzll(i ^ j);
	return d;
}

inline __device__ int sign(int i)
{
	return (i < 0) ? -1 : 1;
}

// TODO: shared memory
__global__ void kernel_buildRadixTree(uint64_t* morton, Node* nodes, int* parents, int numNodes)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;

	// needed to fill parent info:
	// first numLeafs=numNodes+1 element in parents are
	// for leaf nodes. The root index is numLeafs
	const int rootIndex = (numNodes + 1);
	
	if (i >= numNodes) {
		return;
	}
	int t = 0; // while-loop variable

	// determine direction of range
	const int d = sign(delta(i, i+1, morton, numNodes) - delta(i, i-1, morton, numNodes));

	// compute upper bound for the lenght of the range	
	const int delta_min = delta(i, i - d, morton, numNodes);
	int l_max = 128;
	while (delta(i, i+l_max*d, morton, numNodes) > delta_min) {
		l_max *= 4;
	}

	// find the other end with binary search
	int l = 0;
	t = l_max/2;
	do {
		if (delta(i, i+(l+t)*d, morton, numNodes) > delta_min) {
			l += t;
		}
		t /= 2;
	} while (t > 0);
	const int j = i + l*d;
	
	// find split position with binary search
	const int delta_node = delta(i, j, morton, numNodes);
	int s = 0;
	int k = 2;
	t = (l+k-1)/k;
	do {
		if (delta(i, i + (s+t)*d, morton, numNodes) > delta_node) {
			s += t;
		}
		k *= 2;
		t = (l+k-1)/k;
	} while (t > 0);
	const int split = i + s*d + min(d, 0);

	// output child pointers
	Node node;
	node.visited = 0;
	node.left	= min(i, j) == split	? split		: -split;
	node.right	= max(i, j) == split+1	? (split+1)	: -(split+1);

	if (node.left >= 0) {
		parents[node.left] = i;
	} else {
		parents[rootIndex - node.left] = i;
	}
	if (node.right >= 0) {
		parents[node.right] = i;
	} else {
		parents[rootIndex - node.right] = i;
	}

	nodes[i] = node;

	if (i == 0) {
		parents[rootIndex] = -1;
	}
}

// TODO: shared memory
__global__ void kernel_boundingBoxes(Node* nodes, int* parents, int* ids, float3* positions, int numLeafs)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i >= numLeafs) {
		return;
	}

	int parent = parents[i];

	bool finished = false;
	while (!finished) {
		if (atomicAdd(&(nodes[parent].visited), 1) == 0) {
			return;
		} else {
			const int left = nodes[parent].left;
			const int right = nodes[parent].right;
			const float3 bbMinLeft	= left < 0	? nodes[-left].bbMin	: positions[ids[left]];
			const float3 bbMaxLeft	= left < 0	? nodes[-left].bbMax	: positions[ids[left]];
			const float3 bbMinRight = right < 0 ? nodes[-right].bbMin	: positions[ids[right]];
			const float3 bbMaxRight = right < 0 ? nodes[-right].bbMax	: positions[ids[right]];
			nodes[parent].bbMin = fminf(bbMinLeft, bbMinRight); 
			nodes[parent].bbMax = fmaxf(bbMaxLeft, bbMaxRight); 
		}

		parent = parents[numLeafs + parent];
		if (parent == -1) {
			finished = true;
		}
	}
}

BVH::BVH(std::vector<glm::vec3> const& positions, std::vector<glm::vec3> const& normals,
		bool considerNormals)
{
	assert(positions.size() == normals.size());

	if (positions.size() <= 1) {
		std::cout << "not enough points for bvh construction" << std::endl;
		return;
	}
	const int numLeafs = positions.size();
	const int numNodes = numLeafs - 1;
	m_data.positions.resize(numLeafs);
	m_data.morton.resize(numLeafs);
	m_data.ids.resize(numLeafs);
	m_data.parents.resize(numLeafs + numNodes);

	for (int i = 0; i < positions.size(); ++i)
	{
		m_data.positions[i] = make_float3(positions[i]);
	}
	thrust::device_vector<float3> normals_thrust(numLeafs);
	for (int i = 0; i < normals.size(); ++i)
	{
		normals_thrust[i] = make_float3(normals[i]);
	}
	
	cuda::CudaTimer timer;
	timer.start();
	
	thrust::device_vector<float3> normalizedPositions(numLeafs);
	thrust::device_vector<float3> normalizedNormals(numLeafs);

	{ // normalize positions
		float3 min = thrust::reduce(m_data.positions.begin(), m_data.positions.end(), 
				make_float3(std::numeric_limits<float>::max()), Min<float3>());
		float3 max = thrust::reduce(m_data.positions.begin(), m_data.positions.end(),
				make_float3(std::numeric_limits<float>::min()), Max<float3>());
		thrust::transform(m_data.positions.begin(), m_data.positions.end(), normalizedPositions.begin(), Normalize(min, max));
	}
	{ // normalize normals
		float3 min = thrust::reduce(normals_thrust.begin(), normals_thrust.end(), 
				make_float3(std::numeric_limits<float>::max()), Min<float3>());
		float3 max = thrust::reduce(normals_thrust.begin(), normals_thrust.end(),
				make_float3(std::numeric_limits<float>::min()), Max<float3>());
		thrust::transform(normals_thrust.begin(), normals_thrust.end(), normalizedNormals.begin(), Normalize(min, max));
	}
	
	if (considerNormals) {
		thrust::transform(normalizedPositions.begin(), normalizedPositions.end(), 
			normalizedNormals.begin(), m_data.morton.begin(), MortonCode6D());
	} else {
		thrust::transform(normalizedPositions.begin(), normalizedPositions.end(), 
			m_data.morton.begin(), MortonCode3D());
	}

	cuda::CudaTimer timerSort;
	timerSort.start();
	thrust::counting_iterator<int> iter(0);
	thrust::copy(iter, iter + m_data.ids.size(), m_data.ids.begin());
	
	thrust::sort_by_key(m_data.morton.begin(), m_data.morton.end(), m_data.ids.begin());
	
	timerSort.stop();

	m_nodes.resize(numNodes);

	cuda::CudaTimer timerBuildRadixTree;
	timerBuildRadixTree.start();
	{
		dim3 dimBlock(128);
		dim3 dimGrid((numNodes + dimBlock.x - 1) / dimBlock.x);
		kernel_buildRadixTree<<<dimGrid, dimBlock>>>(
			thrust::raw_pointer_cast(&m_data.morton[0]),
			thrust::raw_pointer_cast(&m_nodes[0]),
			thrust::raw_pointer_cast(&m_data.parents[0]),
			numNodes);
	}
	timerBuildRadixTree.stop();

	//std::cout << "start parents: " << std::endl;
	//thrust::copy(m_data.parents.begin(), m_data.parents.end(), std::ostream_iterator<int>(std::cout, "\n"));
	//std::cout << "end parents" << std::endl;
	
	cuda::CudaTimer timerBoundingBoxes;
	timerBoundingBoxes.start();
	{
		dim3 dimBlock(128);
		dim3 dimGrid((numLeafs + dimBlock.x - 1) / dimBlock.x);
		kernel_boundingBoxes<<<dimGrid, dimBlock>>>(
			thrust::raw_pointer_cast(&m_nodes[0]),
			thrust::raw_pointer_cast(&m_data.parents[0]),
			thrust::raw_pointer_cast(&m_data.ids[0]),
			thrust::raw_pointer_cast(&m_data.positions[0]),
			numLeafs);
	}
	timerBoundingBoxes.stop();
	
	timer.stop();
	std::cout << "bvh creation: " << timer.getTime() << std::endl;
	std::cout << "sort: " << timerSort.getTime() << std::endl;
	std::cout << "kernel build radix-tree: " << timerBuildRadixTree.getTime() << std::endl;
	std::cout << "kernel build bounding boxes: " << timerBoundingBoxes.getTime() << std::endl;
	
	// calculate SA of all AABBs
	float sa = 0;
	for (int i = 0; i < m_nodes.size(); ++i) {
		const Node node = m_nodes[i];
		const float3 bbMin = node.bbMin;
		const float3 bbMax = node.bbMax;
		const float dx = abs(bbMax.x - bbMin.x);
		const float dy = abs(bbMax.y - bbMin.y);
		const float dz = abs(bbMax.z - bbMin.z);
	
		sa += 0.001f * 2.f * (dx * (dy + dz) + dy * dz); 
	}
	std::cout << "sum of aabb surface areas: " << sa << std::endl;
	// calculate volume of all AABBs
	float vol = 0;
	for (int i = 0; i < m_nodes.size(); ++i) {
		const Node node = m_nodes[i];
		const float3 bbMin = node.bbMin;
		const float3 bbMax = node.bbMax;
		const float dx = abs(bbMax.x - bbMin.x);
		const float dy = abs(bbMax.y - bbMin.y);
		const float dz = abs(bbMax.z - bbMin.z);
	
		vol += 0.001f * (dx * dy * dz); 
	}
	std::cout << "sum of aabb volumes: " << vol << std::endl;
}

BVH::~BVH()
{
}

void BVH::generateDebugInfo(int level)
{
	thrust::host_vector<Node> nodes = m_nodes;
	thrust::host_vector<int> parents = m_data.parents;
	thrust::host_vector<float3> positions = m_data.positions;

	m_colors.resize(m_data.positions.size());
	m_bbMins.clear();
	m_bbMaxs.clear();

	traverse(m_nodes[0], 0, level);
}

void BVH::traverse(Node const& node, int depth, int level)
{
	if (depth >= level)
	{
		glm::vec3 color = getColor();
		colorChildren(node, color);
		addAABB(node.bbMin, node.bbMax);
	}
	else
	{
		if (node.left < 0) {
			traverse(m_nodes[-node.left], depth + 1, level);
		} else {
			m_colors[m_data.ids[node.left]] = getColor();
			addAABB(m_data.positions[m_data.ids[node.left]],
					m_data.positions[m_data.ids[node.left]]);
		}
		if (node.right < 0) {
			traverse(m_nodes[-node.right], depth + 1, level);
		} else {
			m_colors[m_data.ids[node.right]] = getColor();
			addAABB(m_data.positions[m_data.ids[node.right]],
					m_data.positions[m_data.ids[node.right]]);
		}
	}
}

void BVH::addAABB(float3 const& min, float3 const& max)
{
	const float delta = 5.f;
	glm::vec3 mi = make_vec3(min);
	glm::vec3 ma = make_vec3(max);
	if (glm::length(mi.x - ma.x) < 0.1f) {
		mi -= glm::vec3(delta, 0.f, 0.f);
		ma += glm::vec3(delta, 0.f, 0.f);
	}
	if (glm::length(mi.y - ma.y) < 0.1f) {
		mi -= glm::vec3(0.f, delta, 0.f);
		ma += glm::vec3(0.f, delta, 0.f);
	}
	if (glm::length(mi.z - ma.z) < 0.1f) {
		mi -= glm::vec3(0.f, 0.f, delta);
		ma += glm::vec3(0.f, 0.f, delta);
	}
	m_bbMins.push_back(mi);
	m_bbMaxs.push_back(ma);
}

void BVH::colorChildren(Node const& node, glm::vec3 const& color)
{
	if (node.left < 0) {
		colorChildren(m_nodes[-node.left], color);
	} else {
		m_colors[m_data.ids[node.left]] = color;
	}
	if (node.right < 0) {
		colorChildren(m_nodes[-node.right], color);
	} else {
		m_colors[m_data.ids[node.right]] = color;
	}
}

glm::vec3 BVH::getColor()
{
	static int color = -1;
	color++;
	switch(color % 10){
		case 0: return glm::vec3(0.8f, 0.8f, 0.8f);
		case 1: return glm::vec3(0.0f, 0.0f, 0.5f);
		case 2: return glm::vec3(0.0f, 0.5f, 0.0f);
		case 3: return glm::vec3(0.0f, 0.5f, 0.5f);
		case 4: return glm::vec3(1.0f, 0.0f, 0.0f);
		case 5: return glm::vec3(1.0f, 0.0f, 0.5f);
		case 6: return glm::vec3(1.0f, 0.5f, 0.0f);
		case 7: return glm::vec3(1.0f, 0.5f, 0.5f);
		case 8: return glm::vec3(0.5f, 0.5f, 0.0f);
		case 9: return glm::vec3(0.5f, 0.0f, 0.5f);
		default: return glm::vec3(0.f);
	}
}
