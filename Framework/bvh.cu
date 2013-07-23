#include "bvh.h"
#include "Utils/stream.h"
#include "morton.h"

#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/scan.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>
#include <thrust/host_vector.h>
#include <thrust/random/linear_congruential_engine.h>
#include <thrust/random/uniform_real_distribution.h>

#include <limits>
#include <random>

#include "CudaResources/cudaUtil.hpp"
#include "CudaResources/cudaTimer.hpp"

#include <intrin.h>

#pragma intrinsic(_BitScanReverse)
#pragma intrinsic(_BitScanReverse64)

//#define PRINT_DEBUG

std::mt19937 rng;
std::uniform_real_distribution<float> dist01;

template<typename T>
void printBinary(T i)
{
	T one = 1;
	T shift = sizeof(T) * 8 - 1;
	for (T z = (one << shift); z > 0; z >>= 1) {
		if ((i & z) == z) { std::cout << "1"; }
		else { std::cout << "0"; }
	}
	std::cout << std::endl;
}

template<typename T>
void printVector(std::string const& text, std::vector<T> const& vector)
{
	std::cout << text << " ";
	for (int i = 0; i < vector.size(); i++) {
		std::cout << vector[i] << ", ";
	}
	std::cout << std::endl;
}

template<typename T>
void printDeviceVector(std::string const& text, thrust::device_vector<T> const& vector)
{
	std::cout << text << " ";
	for (int i = 0; i < vector.size(); i++) {
		std::cout << vector[i] << ", ";
	}
	std::cout << std::endl;
}

template<typename T>
void printMortonCodes(thrust::device_vector<T> const& vector)
{
	std::cout << "morton codes: " << std::endl;
	for (int i = 0; i < vector.size(); i++) {
		T val = vector[i];
		printBinary<T>(val);
	}
	std::cout << "end morton codes" << std::endl;
}

struct Normalize {
	Normalize(float3 const& min, float3 const& max) : m_min(min), m_max(max) {}

	__host__ __device__ float3 operator() (float3 const& v) {
		return (v - m_min) / (m_max - m_min);
	}

	float3 m_min, m_max;
};

template<typename T>
struct add_mul : public thrust::binary_function<T,T,T>
{
	add_mul(const T& _factor) : factor(_factor) {
	}
		__host__ __device__ T operator()(const T& a, const T& b) const 
	{
		return (a + b * factor);
	}
	const T factor;
};

template<typename T>
struct Max : public thrust::binary_function<T,T,T> {
	__host__ __device__ T operator()(const T& a, const T& b) const {
		return max(a, b);
	}
};

template<typename T>
struct Min : public thrust::binary_function<T,T,T> {
	__host__ __device__ T operator()(const T& a, const T& b) const {
		return min(a, b);
	}
};

template<typename T>
struct Center : public thrust::binary_function<T,T,T> {
	__host__ __device__ T operator()(const T& a, const T& b) const {
		return (a + b) / 2;
	}
};

inline __device__ int cuda_clz(uint32_t x) {
	return __clz(x);
}

inline __device__ int cuda_clz(uint64_t x) {
	return __clzll(x);
}

template<typename T>
inline __device__ int cuda_delta(int i, int j, T* morton, int numNodes)
{
	if (j < 0 || j > numNodes) {
		return -1;
	}
	int d = cuda_clz(morton[i] ^ morton[j]); 
	if (d==sizeof(T)*8) {
		d += cuda_clz(uint32_t(i ^ j));
	}
	return d;
}

inline __host__ int cpu_clz(uint32_t x) {
	if (x == 0) return 32;
	unsigned long r = 0;
	_BitScanReverse(&r, x);
	return 31 - r;
}

inline __host__ int cpu_clz(uint64_t x) {
	if (x == 0) return 64;
	unsigned long r = 0;
	_BitScanReverse64(&r, x);
	return 63 - r;
}

template<typename T>
inline __host__ int cpu_delta(int i, int j, T* morton, int numNodes)
{
	if (j < 0 || j > numNodes) {
		return -1;
	}
	int d = cpu_clz(morton[i] ^ morton[j]); 
	if (d==sizeof(T)*8) {
		d += cpu_clz(uint32_t(i ^ j));
	}
	return d;
}

#ifdef __CUDA_ARCH__
#define delta cuda_delta
#else
#define delta cpu_delta
#endif

inline __host__ __device__ int sign(int i)
{
	return (i < 0) ? -1 : 1;
}

template<typename T>
__global__ void kernel_clz(T* morton, int* clz_res, int numMortonCodes) {
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	const int j = blockIdx.y * blockDim.y + threadIdx.y;

	if (i >= numMortonCodes || j >= numMortonCodes)
		return;

	int index = i + j * numMortonCodes;

	clz_res[index] = delta(i, j, morton, numMortonCodes);
}

// build the i-th inner node of the radix tree
template<typename T>
__host__ __device__ void buildRadixTree(int i, BvhNode* nodes, T* morton, int* parents, int numNodes)
{
	// needed to fill parent info:
	// first numLeafs=numNodes+1 element in parents are
	// for leaf nodes. The root index is numLeafs
	const int rootIndex = (numNodes + 1);
	
	if (i >= numNodes) {
		return;
	}
	int t = 0; // while-loop variable

	// determine direction of range
	const int d = sign(delta(i, i+1, morton, numNodes) 
		- delta(i, i-1, morton, numNodes));

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
	while (true) {
		if (delta(i, i + (s+t)*d, morton, numNodes) > delta_node) {
			s += t;
		}
		
		if (t <= 1) { break; }
		
		k *= 2;
		t = (l+k-1)/k;
	}
	const int split = i + s*d + min(d, 0);

	// output child pointers
	BvhNode node;
	node.visited = 0;
	node.left	= min(i, j) == split	? split		: -(split+1);
	node.right	= max(i, j) == split+1	? (split+1)	: -(split+2);

	if (node.left >= 0) {
		parents[node.left] = i;
	} else {
		const int index = rootIndex - (node.left+1);
		parents[index] = i;
	}
	if (node.right >= 0) {
		const int index = node.right;
		parents[index] = i;
	} else {
		const int index = rootIndex - (node.right+1);
		parents[index] = i;
	}

	nodes[i] = node;

	if (i == 0) {
		parents[rootIndex] = -1;
	}
}

template<typename T>
__global__ void kernel_buildRadixTree(BvhNode* nodes, T* morton, int* parents, int numNodes)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
		
	if (i >= numNodes) {
		return;
	}

	buildRadixTree(i, nodes, morton, parents, numNodes);
}

template<typename NodeDataParam>
__global__ void kernel_innerNodes(BvhNode* nodes, int* parents, 
	float3* positions, int numLeafs, NodeDataParam* param)
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
			const float3 bbMinLeft	= left < 0	? nodes[-(left+1)].bbMin	: positions[left];
			const float3 bbMaxLeft	= left < 0	? nodes[-(left+1)].bbMax	: positions[left];
			const float3 bbMinRight = right < 0 ? nodes[-(right+1)].bbMin	: positions[right];
			const float3 bbMaxRight = right < 0 ? nodes[-(right+1)].bbMax	: positions[right];
			nodes[parent].bbMin = fminf(bbMinLeft, bbMinRight); 
			nodes[parent].bbMax = fmaxf(bbMaxLeft, bbMaxRight); 

			if (param != 0) {
				cluster(parent, left, right, numLeafs, param);
			}
		}

		parent = parents[numLeafs + parent];
		if (parent == -1) {
			finished = true;
		}
	}
}

Bvh::Bvh(bool considerNormals)
	: m_considerNormals(considerNormals)
{ }

Bvh::~Bvh()
{ }

void Bvh::create()
{
	if (m_input->positions.size() <= 1) {
		std::cout << "not enough points for bvh construction" << std::endl;
		return;
	}
	m_numLeafs = m_input->positions.size();
	const int numLeafs = m_input->positions.size();
	const int numNodes = numLeafs - 1;
	m_data.reset(new BvhData());
	m_data->morton.resize(numLeafs);
	m_data->parents.resize(numLeafs + numNodes);
	m_data->numLeafs = numLeafs;
	m_data->numNodes = numNodes;
	
	cuda::CudaTimer timer;
	timer.start();
	
	thrust::device_vector<float3> normalizedPositions(numLeafs);
	thrust::device_vector<float3> normalizedNormals(numLeafs);

	{ // normalize positions
		float3 min = thrust::reduce(m_input->positions.begin(), m_input->positions.end(), 
			make_float3(std::numeric_limits<float>::max()), Min<float3>());
		float3 max = thrust::reduce(m_input->positions.begin(), m_input->positions.end(),
			make_float3(std::numeric_limits<float>::min()), Max<float3>());
		thrust::transform(m_input->positions.begin(), m_input->positions.end(), 
			normalizedPositions.begin(), Normalize(min, max));
	}
	{ // normalize normals
		float3 min = thrust::reduce(m_input->normals.begin(), m_input->normals.end(), 
			make_float3(std::numeric_limits<float>::max()), Min<float3>());
		float3 max = thrust::reduce(m_input->normals.begin(), m_input->normals.end(),
			make_float3(std::numeric_limits<float>::min()), Max<float3>());
		thrust::transform(m_input->normals.begin(), m_input->normals.end(), 
			normalizedNormals.begin(), Normalize(min, max));
	}
	
	if (m_considerNormals) {
		thrust::transform(normalizedPositions.begin(), normalizedPositions.end(), 
			normalizedNormals.begin(), m_data->morton.begin(), MortonCode6D());
	} else {
		thrust::transform(normalizedPositions.begin(), normalizedPositions.end(), 
			m_data->morton.begin(), MortonCode3D());
	}

	cuda::CudaTimer timerSort;
	timerSort.start();

	sort();
	timerSort.stop();
	m_nodes.resize(numNodes);
	
	//{ // for debug purposes: check morton clz
	//	int numMortonCodes = m_data->morton.size();
	//	thrust::device_vector<int> clz_cuda(numMortonCodes * numMortonCodes);
	//	dim3 dimBlock(16, 16);
	//	dim3 dimGrid((numMortonCodes + dimBlock.x - 1)/dimBlock.x, (numMortonCodes + dimBlock.y - 1)/dimBlock.y);
	//	kernel_clz<<<dimGrid, dimBlock>>>(
	//		thrust::raw_pointer_cast(&m_data->morton[0]), 
	//		thrust::raw_pointer_cast(&clz_cuda[0]),
	//		numMortonCodes
	//		);
	//	std::vector<MC_SIZE> morton_cpu(numLeafs);
	//	thrust::copy(m_data->morton.begin(), m_data->morton.end(), morton_cpu.begin());
	//	for (int i = 0; i < numMortonCodes; ++i) {
	//		for (int j = 0; j < numMortonCodes; ++j) {
	//			int index = i + j * numMortonCodes;
	//			int clz_ref = cpu_delta(i, j, morton_cpu.data(), numMortonCodes);
	//			if (clz_ref != clz_cuda[index]) {
	//				std::cout << "(i,j)=(" << i << "," << j << ")" << std::endl;
	//				std::cout << "    clz_cuda = " << clz_cuda[index] << std::endl;
	//				std::cout << "    clz_ref = " << clz_ref << std::endl;
	//			}
	//		}
	//	}
	//}
	
	//{ // for debug purposes: construct radix tree on cpu
	//	std::vector<BvhNode> nodes(numNodes);
	//	std::vector<MC_SIZE> morton(numLeafs);
	//	thrust::copy(m_data->morton.begin(), m_data->morton.end(), morton.begin());
	//	std::vector<int> parents(numNodes + numLeafs);
	//	for (int i = 0; i < numNodes; ++i) {
	//		buildRadixTree(i, nodes.data(), morton.data(), parents.data(), numNodes);
	//	}
	//	checkTreeIntegrity(nodes, parents);
	//}
	
	cuda::CudaTimer timerBuildRadixTree;
	timerBuildRadixTree.start();
	{
		dim3 dimBlock(128);
		dim3 dimGrid((numNodes + dimBlock.x - 1) / dimBlock.x);
		kernel_buildRadixTree<<<dimGrid, dimBlock>>>(
			thrust::raw_pointer_cast(&m_nodes[0]), 
			thrust::raw_pointer_cast(&m_data->morton[0]), 
			thrust::raw_pointer_cast(&m_data->parents[0]), 
			m_data->numNodes);
	}
	timerBuildRadixTree.stop();
	
	//{ // for debug purposes: check the integrity of the radix tree
	//	std::vector<int> parents(m_data->parents.size());
	//	thrust::copy(m_data->parents.begin(), m_data->parents.end(), parents.begin());
	//	m_nodesDebug.resize(m_nodes.size());
	//	thrust::copy(m_nodes.begin(), m_nodes.end(), m_nodesDebug.begin());
	//	checkTreeIntegrity(m_nodesDebug, parents);
	//	printDebugRadixTree();
	//}

	cuda::CudaTimer timerFillInnerNodes;
	timerFillInnerNodes.start();
	fillInnerNodes();
	timerFillInnerNodes.stop();

	BvhParam param;
	param.numLeafs = numLeafs;
	param.numNodes = numNodes;
	param.nodes = thrust::raw_pointer_cast(&m_nodes[0]);
	param.positions = thrust::raw_pointer_cast(&m_input->positions[0]);
	param.normals = thrust::raw_pointer_cast(&m_input->normals[0]);
	m_param.reset(new cuda::CudaBuffer<BvhParam>(&param));
	
	timer.stop();
	//std::cout << "bvh creation: " << timer.getTime() << std::endl;
	//std::cout << "sort: " << timerSort.getTime() << std::endl;
	//std::cout << "kernel build radix-tree: " << timerBuildRadixTree.getTime() << std::endl;
	//std::cout << "kernel build inner nodes: " << timerFillInnerNodes.getTime() << std::endl;
}

void Bvh::printDebugRadixTree()
{
	int numLeafs = m_input->positions.size();
	std::cout << "traverse radix tree: (" << numLeafs << " leafs)" << std::endl;
	
	int stack[64];
	memset(stack, 0, 64 * sizeof(int));
	stack[0] = -1;
	int stack_ptr = 1;

	while (stack_ptr > 0) 
	{
		stack_ptr--;
		const int nodeIndex = stack[stack_ptr];

		if (nodeIndex >= 0) { // leaf
			std::cout << "    Childnode " << nodeIndex << std::endl;
			std::cout << "        parent: " << m_data->parents[nodeIndex] << std::endl;
		} else { // inner node
			const BvhNode bvhNode = m_nodes[-(nodeIndex+1)];
			std::cout << "    Node " << nodeIndex << std::endl;
			std::cout << "        parent: " << m_data->parents[numLeafs + -(nodeIndex+1)] << std::endl;
			std::cout << "        left: " << bvhNode.left << std::endl;
			std::cout << "        right: " << bvhNode.right << std::endl;

			stack[stack_ptr] = bvhNode.left;
			stack_ptr++;
			stack[stack_ptr] = bvhNode.right;
			stack_ptr++;
		}
	}
}

void Bvh::generateDebugInfo(int level)
{	
	m_positions.clear();
	m_colors.clear();
	m_positions.resize(m_data->morton.size());
	for (int i = 0; i < m_data->morton.size(); ++i) {
		float3 pos = m_input->positions[i];
		m_positions[i] = glm::vec3(pos.x, pos.y, pos.z);
	}
	m_colors.resize(m_data->morton.size());
	m_bbMins.clear();
	m_bbMaxs.clear();

	m_nodesDebug.resize(m_nodes.size());
	thrust::copy(m_nodes.begin(), m_nodes.end(), m_nodesDebug.begin());
	
	traverse(m_nodesDebug[0], 0, level);
}

void Bvh::traverse(BvhNode const& node, int depth, int level)
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
			traverse(m_nodesDebug[-(node.left+1)], depth + 1, level);
		} else {
			m_colors[node.left] = getColor();
			addAABB(m_input->positions[node.left], m_input->positions[node.left]);
		}
		if (node.right < 0) {
			traverse(m_nodesDebug[-(node.right+1)], depth + 1, level);
		} else {
			m_colors[node.right] = getColor();
			addAABB(m_input->positions[node.right], m_input->positions[node.right]);
		}
	}
}

void Bvh::addAABB(float3 const& min, float3 const& max)
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

void Bvh::colorChildren(BvhNode const& node, glm::vec3 const& color)
{
	if (node.left < 0) {
		colorChildren(m_nodesDebug[-(node.left+1)], color);
	} else {
		m_colors[node.left] = color;
	}
	if (node.right < 0) {
		colorChildren(m_nodesDebug[-(node.right+1)], color);
	} else {
		m_colors[node.right] = color;
	}
}

void Bvh::checkTreeIntegrity(std::vector<BvhNode> const& nodes, std::vector<int> const& parents)
{
	std::vector<int> stack(4096);
	int numNodes = nodes.size();
	int numLeafs = numNodes + 1;
	stack[0] = -1;
	int stack_ptr = 1;
	int max_stack_ptr = 0;
	std::vector<int> checkparent(numNodes);
	std::vector<int> visited(numNodes + numLeafs);
	std::fill(checkparent.begin(), checkparent.end(), 0);
	std::fill(visited.begin(), visited.end(), 0);

	while (stack_ptr > 0) {
		int nodeIndex = stack[--stack_ptr];

		if (nodeIndex >= 0) { // leaf
			if (nodeIndex >= numLeafs) {
				std::cout << "leaf node index " << nodeIndex << " out of bounds";
			}
			int parent = parents[nodeIndex];
			if (parent != -1) { // if not root
				checkparent[parent]++;
			}

			if (visited[nodeIndex] == 1) {
				std::cout << "node " << nodeIndex << " already visited!" << std::endl;
				printDeviceVector("positions: ", m_input->positions);
				return;
			} else {
				visited[nodeIndex] = 1;
			}
		}
		else { // inner node
			if (-(nodeIndex+1) >= numNodes) {
				std::cout << "leaf node index " << nodeIndex << " out of bounds";
			}
			BvhNode node = nodes[-(nodeIndex+1)];
			int idx = numLeafs - (nodeIndex+1);
			int parent = parents[idx];
			if (parent != -1) { // if not root
				checkparent[parent]++;
			}

			if (visited[idx] == 1) {
				std::cout << "node " << nodeIndex << " already visited!" << std::endl;
				printDeviceVector("positions: ", m_input->positions);
				return;
			} else {
				visited[idx] = 1;
			}

			stack[stack_ptr++] = node.left;
			stack[stack_ptr++] = node.right;
			if (stack_ptr > max_stack_ptr) {
				max_stack_ptr = stack_ptr;
			}
		}
	}

	for (int i = 0; i < checkparent.size(); ++i) {
		if (checkparent[i] != 2) {
			std::cout << "node " << i << " only has one child" << std::endl;
		}
	}
	for (int i = 0; i < visited.size(); ++i) {
		if (visited[i] != 1) {
			std::cout << "nodechild " << i << " was not visited" << std::endl;
		}
	}
	std::cout << "max stack_ptr: " << max_stack_ptr << std::endl;
}

glm::vec3 Bvh::getColor()
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

BvhNode Bvh::getNode(int i)
{
	if (i < 0 || i >= m_nodes.size()) {
		std::cout << "index out of range" << std::endl;
		return m_nodes[0];
	}
	return m_nodes[i];
}

// ----------------------------------------------------------------------
// AvplBvh --------------------------------------------------------------
// ----------------------------------------------------------------------
	
AvplBvhNodeData::AvplBvhNodeData(std::vector<AVPL> const& avpls)
{
	const int numLeafs = avpls.size();
	const int numElements = 2 * numLeafs - 1;

	std::vector<int> temp_size(numLeafs);
	std::vector<int> temp_materialIndex(numLeafs);
	std::vector<float3> temp_position(numLeafs);
	std::vector<float3> temp_normal(numLeafs);
	std::vector<float3> temp_incRadiance(numLeafs);
	std::vector<float3> temp_incDirection(numLeafs);

	for (int i = 0; i < numLeafs; ++i) {
		temp_size[i] = 1;
		temp_materialIndex[i] = avpls[i].m_MaterialIndex;
		temp_position[i] = make_float3(avpls[i].GetPosition());
		temp_normal[i] = make_float3(avpls[i].GetOrientation());
		temp_incRadiance[i] = make_float3(avpls[i].GetIncidentRadiance());
		temp_incDirection[i] = make_float3(avpls[i].m_Direction);
	}
	
	std::vector<float> temp_rand(numElements);
	for (int i = 0; i < numElements; ++i) {
		temp_rand[i] = dist01(rng);
	}
	
	size.resize(numElements);
	materialIndex.resize(numElements);
	position.resize(numElements);
	normal.resize(numElements);
	incRadiance.resize(numElements);
	incDirection.resize(numElements);
	randomNumbers.resize(numElements);

	thrust::copy(temp_size.begin(), temp_size.end(), size.begin());
	thrust::copy(temp_materialIndex.begin(), temp_materialIndex.end(), materialIndex.begin());
	thrust::copy(temp_position.begin(), temp_position.end(), position.begin());
	thrust::copy(temp_normal.begin(), temp_normal.end(), normal.begin());
	thrust::copy(temp_incRadiance.begin(), temp_incRadiance.end(), incRadiance.begin());
	thrust::copy(temp_incDirection.begin(), temp_incDirection.end(), incDirection.begin());
	thrust::copy(temp_rand.begin(), temp_rand.end(), randomNumbers.begin());

	AvplBvhNodeDataParam param;
	param.size = thrust::raw_pointer_cast(&size[0]), 
	param.materialIndex = thrust::raw_pointer_cast(&materialIndex[0]), 
	param.randomNumbers = thrust::raw_pointer_cast(&randomNumbers[0]), 
	param.position = thrust::raw_pointer_cast(&position[0]), 
	param.normal = thrust::raw_pointer_cast(&normal[0]), 
	param.incRadiance = thrust::raw_pointer_cast(&incRadiance[0]), 
	param.incDirection = thrust::raw_pointer_cast(&incDirection[0]), 
	m_param.reset(new cuda::CudaBuffer<AvplBvhNodeDataParam>(&param));
}

__device__ __host__ void cluster(const int target, const int left, const int right, int numLeafs, AvplBvhNodeDataParam* param)
{
	const int targetId = numLeafs + target;
	const int leftId = left < 0 ? numLeafs - (left+1) : left;
	const int rightId = right < 0 ? numLeafs - (right+1) : right;

	const float i1 = length(param->incRadiance[leftId]);
	const float i2 = length(param->incRadiance[rightId]);
	int repr = leftId;
	if (param->randomNumbers[targetId]> (i1 / (i1 + i2)))
		repr = rightId;

	param->size[targetId] = param->size[leftId] + param->size[rightId];
	param->materialIndex[targetId] = param->materialIndex[repr];
	param->incRadiance[targetId] = param->incRadiance[leftId] + param->incRadiance[rightId];
	param->incDirection[targetId] = param->incDirection[repr];
	param->position[targetId] = param->position[repr];
	param->normal[targetId] = param->normal[repr];
}

AvplBvh::AvplBvh(std::vector<AVPL> const& avpls, bool considerNormals)
	: Bvh(considerNormals)
{
	m_input.reset(new BvhInput());

	std::vector<float3> pos;
	std::vector<float3> norm;
	for (int i = 0; i < avpls.size(); ++i)
	{
		if (avpls[i].GetBounce() > 0) {
			pos.push_back(make_float3(avpls[i].GetPosition()));
			norm.push_back(make_float3(avpls[i].GetOrientation()));
		}
	}
	
	if (pos.size() <= 1) {
		std::cout << "not enough points for bvh construction" << std::endl;
		return;
	}

	m_input->positions.resize(pos.size());
	m_input->normals.resize(pos.size());
	thrust::copy(pos.begin(), pos.end(), m_input->positions.begin());
	thrust::copy(norm.begin(), norm.end(), m_input->normals.begin());

	m_nodeData.reset(new AvplBvhNodeData(avpls));

	create();
}


AvplBvh::~AvplBvh()
{
}

void AvplBvh::fillInnerNodes()
{
	dim3 dimBlock(128);
	dim3 dimGrid((m_numLeafs + dimBlock.x - 1) / dimBlock.x);
	kernel_innerNodes<AvplBvhNodeDataParam><<<dimGrid, dimBlock>>>(
		thrust::raw_pointer_cast(&m_nodes[0]), 
		thrust::raw_pointer_cast(&m_data->parents[0]), 
		thrust::raw_pointer_cast(&m_input->positions[0]), 
		m_data->numLeafs,
		m_nodeData->m_param->getDevicePtr());
}

void AvplBvh::sort()
{
	thrust::sort_by_key(m_data->morton.begin(), m_data->morton.end(),
		thrust::make_zip_iterator(thrust::make_tuple(
				m_input->positions.begin(), 
				m_input->normals.begin(),
				m_nodeData->size.begin(),
				m_nodeData->materialIndex.begin(),
				m_nodeData->position.begin(),
				m_nodeData->normal.begin(),
				m_nodeData->incRadiance.begin(),
				m_nodeData->incDirection.begin()
			)	
		)
	);
}


void AvplBvh::testTraverse()
{
#ifdef PRINT_DEBUG
	int numLeafs = m_input->positions.size(); 
	std::vector<int> leafVisited(numLeafs);
	std::fill(leafVisited.begin(), leafVisited.end(), 0);

	std::cout << "traverse radix tree: (" << numLeafs << " leafs)" << std::endl;
	
	int stack[64];
	memset(stack, 0, 64 * sizeof(int));
	stack[0] = -1;
	int stack_ptr = 1;

	while (stack_ptr > 0) 
	{
		stack_ptr--;
		const int nodeIndex = stack[stack_ptr];

		if (nodeIndex >= 0) { // leaf
			int idx = nodeIndex;
			std::cout << "    Childnode " << nodeIndex << std::endl;
			std::cout << "        pos: " << m_nodeData->position[idx] << std::endl;
			std::cout << "        norm: " << m_nodeData->normal[idx] << std::endl;
			std::cout << "        size: " << m_nodeData->size[idx] << std::endl;
			std::cout << "        incRad: " << m_nodeData->incRadiance[idx] << std::endl;
			std::cout << "        incDir: " << m_nodeData->incDirection[idx] << std::endl;
		} else { // inner node
			int idx = numLeafs-(nodeIndex+1);
			const BvhNode bvhNode = m_nodes[-(nodeIndex+1)];
			std::cout << "    Node " << nodeIndex << std::endl;
			std::cout << "        left: " << bvhNode.left << std::endl;
			std::cout << "        right: " << bvhNode.right << std::endl;
			std::cout << "        pos: " << m_nodeData->position[idx] << std::endl;
			std::cout << "        norm: " << m_nodeData->normal[idx] << std::endl;
			std::cout << "        size: " << m_nodeData->size[idx] << std::endl;
			std::cout << "        incRad: " << m_nodeData->incRadiance[idx] << std::endl;
			std::cout << "        incDir: " << m_nodeData->incDirection[idx] << std::endl;
			stack[stack_ptr] = bvhNode.left;
			stack_ptr++;
			stack[stack_ptr] = bvhNode.right;
			stack_ptr++;
		}
	}
#endif
}


SimpleBvh::SimpleBvh(std::vector<float3> const& positions,
		std::vector<float3> const& normals,
		bool considerNormals)
	: Bvh(considerNormals)
{
	m_input.reset(new BvhInput());

	if (positions.size() <= 1) {
		std::cout << "not enough points for bvh construction" << std::endl;
		return;
	}

	m_input->positions.resize(positions.size());
	m_input->normals.resize(positions.size());
	thrust::copy(positions.begin(), positions.end(), m_input->positions.begin());
	thrust::copy(normals.begin(), normals.end(), m_input->normals.begin());

	create();
}

// ----------------------------------------------------------------------
// SimpleBvh --------------------------------------------------------------
// ----------------------------------------------------------------------

SimpleBvh::~SimpleBvh()
{ }

void SimpleBvh::sort()
{
	thrust::sort_by_key(m_data->morton.begin(), m_data->morton.end(),
		thrust::make_zip_iterator(thrust::make_tuple(
				m_input->positions.begin(), 
				m_input->normals.begin()
			)	
		)
	);
}

void SimpleBvh::fillInnerNodes()
{
	dim3 dimBlock(128);
	dim3 dimGrid((m_numLeafs + dimBlock.x - 1) / dimBlock.x);
	kernel_innerNodes<AvplBvhNodeDataParam><<<dimGrid, dimBlock>>>(
		thrust::raw_pointer_cast(&m_nodes[0]), 
		thrust::raw_pointer_cast(&m_data->parents[0]), 
		thrust::raw_pointer_cast(&m_input->positions[0]), 
		m_data->numLeafs,
		0);
}

// ----------------------------------------------------------------------
// VisiblePointsBvh --------------------------------------------------------------
// ----------------------------------------------------------------------

typedef thrust::zip_iterator<
			thrust::tuple<
				thrust::device_vector<int>::iterator, 
				thrust::device_vector<int>::iterator
			>
		> ZipIterator;

struct Predicate {
	Predicate(int *bucketDelimiters) : m_bucketDelimiters(bucketDelimiters) {}

	__host__ __device__
	bool operator()(thrust::tuple<int, int> const& a, thrust::tuple<int, int> const& b) {
		if (a.get<0>() == b.get<0>()) {
 			return true;
		} else {
 			m_bucketDelimiters[b.get<1>()] = 1;
			return false;
		}
	}

	int* m_bucketDelimiters;
};

struct Mapping {
	Mapping(int *mapping) : m_mapping(mapping) {}

	__host__ __device__
	void operator()(thrust::tuple<int, int> const& a) {
		m_mapping[a.get<0>()] = a.get<1>();
	}

	int* m_mapping;
};

struct MaxFloat3 {
	__host__ __device__ float3 operator()(float3 const& a, float3 const& b) const {
		return max(a, b);
	}
};

struct MinFloat3 {
	__host__ __device__ float3 operator()(float3 const& a, float3 const& b) const {
		return min(a, b);
	}
};

__device__ __host__ void cluster(const int target, const int left, const int right, int numLeafs, VisiblePointsBvhNodeDataParam* param)
{
}

__global__ void kernel_fillVisiblePoints(
	cudaSurfaceObject_t inPositions, cudaSurfaceObject_t inNormals,
	float3* positions, float3* normals, uint2* pixels, int width, int height)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= width || y >= height) {
		return;
	}
	
	float4 data;
	surf2Dread(&data, inPositions, x * sizeof(float4), y);
	const float3 pos = make_float3(data);
	surf2Dread(&data, inNormals, x * sizeof(float4), y);
	const float3 norm = make_float3(data);

	int index = x + y * width;
	positions[index] = pos;
	normals[index] = norm;
	pixels[index] = make_uint2(x, y);	
}

__global__ void kernel_initBuffers(dim3 dimensions, int* tileIdBuffer, int* pixelIdBuffer, int* bucketBuffer,
		int* tempClusterIdBuffer, int* clusterIdBuffer, int* cluster_k, float3* positions,
		float3 cameraPos, float zNear, float cameraTheta, cudaSurfaceObject_t positionTexture)
{
	const int x = threadIdx.x + blockIdx.x * blockDim.x;
	const int y = threadIdx.y + blockIdx.y * blockDim.y;
	
	if (x >= dimensions.x || y >= dimensions.y)	{
		return;
	}

	const int offset = x + y * dimensions.x;
	const int tileId = blockIdx.y * gridDim.x + blockIdx.x;
	
	tileIdBuffer[offset] = tileId;
	pixelIdBuffer[offset] = offset;
	bucketBuffer[offset] = 0;
	tempClusterIdBuffer[offset] = 0;
	clusterIdBuffer[offset] = 0;

	float4 temp;
	surf2Dread(&temp, positionTexture, x * sizeof(float4), y);

	const float3 position = make_float3(temp);
	const float dist = length(cameraPos - position);

	const int k = floor( 
		log(dist / zNear) / 
		log(1.f + (2.f * tan(cameraTheta)) / blockDim.y) );
	
	cluster_k[offset] = k;
	positions[offset] = position;
}

VisiblePointsBvh::VisiblePointsBvh(cudaSurfaceObject_t positionSurfaceObject,
		cudaSurfaceObject_t normalSurfaceObject, int width, int height, bool considerNormals,
		float3 const& cameraPos, float zNear, float cameraTheta)
	: Bvh(considerNormals)
{
	m_input.reset(new BvhInput());

	const int bufferSize = width * height;
	const int tileWidth = 16;
	const int tileHeight = 16;

	const int numTilesX = (width  + tileWidth  - 1) / tileWidth;
	const int numTilesY = (height + tileHeight - 1) / tileHeight;
	
	dim3 dimensions(width, height, 1);
	dim3 numBlocks(numTilesX, numTilesY, 1);
	dim3 numThreads(tileWidth, tileHeight, 1);

	thrust::device_vector<int> tileIdBuffer(bufferSize);
	thrust::device_vector<int> pixelIdBuffer(bufferSize);
	thrust::device_vector<int> bucketBuffer(bufferSize);
	thrust::device_vector<int> tempClusterIdBuffer(bufferSize);
	thrust::device_vector<int> clusterMinIdBuffer(bufferSize);
	thrust::device_vector<int> clusterMaxIdBuffer(bufferSize);
	thrust::device_vector<float3> positions(bufferSize);
	thrust::device_vector<int> cluster_k(bufferSize);
	
	m_clusterBBMin.clear();
	m_clusterBBMax.clear();
	m_clusterIdBuffer.clear();
	m_clusterBBMin.resize(bufferSize);
	m_clusterBBMax.resize(bufferSize);
	m_clusterIdBuffer.resize(bufferSize);
	
	kernel_initBuffers<<<numBlocks, numThreads>>>(dimensions
		, (int*) thrust::raw_pointer_cast(&tileIdBuffer[0])
		, (int*) thrust::raw_pointer_cast(&pixelIdBuffer[0])
		, (int*) thrust::raw_pointer_cast(&bucketBuffer[0])
		, (int*) thrust::raw_pointer_cast(&tempClusterIdBuffer[0])
		, (int*) thrust::raw_pointer_cast(&m_clusterIdBuffer[0])
		, (int*) thrust::raw_pointer_cast(&cluster_k[0])
		, (float3*) thrust::raw_pointer_cast(&positions[0])
		, cameraPos, zNear, cameraTheta
		, positionSurfaceObject
	);

	// determine the maximal depth value of all clusters
	const int maxValue = *(thrust::max_element(
		cluster_k.begin(), cluster_k.end()));

	// transform the values of each tile block into its owm value domain 
	thrust::transform(cluster_k.begin(), cluster_k.end(), 
		tileIdBuffer.begin(), cluster_k.begin(), add_mul<int>(maxValue)); 
	
	// sort the transformed values and keep track of the corresponding
	// tile and pixel id
	thrust::sort_by_key(cluster_k.begin(), cluster_k.end(), 
		thrust::make_zip_iterator(
			thrust::make_tuple(
				tileIdBuffer.begin(),
				pixelIdBuffer.begin(),
				positions.begin())),
		thrust::less<int>());

	// remove duplicate values and indicate the places where a new bucket starts
	thrust::counting_iterator<int> iter(0);
	thrust::device_vector<int> ids(bufferSize);
	thrust::copy(iter, iter + ids.size(), ids.begin());
	ZipIterator source_begin(thrust::make_tuple(
		cluster_k.begin(), ids.begin()));
	ZipIterator source_end(thrust::make_tuple(
		cluster_k.end(), ids.end()));
	
	thrust::pair<ZipIterator, thrust::device_vector<int>::iterator> new_end = 
		thrust::unique_by_key(source_begin, source_end, tileIdBuffer.begin(),
			Predicate(thrust::raw_pointer_cast(&bucketBuffer[0])));
	
	// calculate the prefix sum of the bucket delimiters to determine the 
	// "cluster-id" at the corresponding index
	thrust::inclusive_scan(bucketBuffer.begin(), bucketBuffer.end(), 
		tempClusterIdBuffer.begin());
	
	// determine AABB of the clusters in world space 
	thrust::pair<thrust::device_vector<int>::iterator, thrust::device_vector<float3>::iterator> end_max = 
		thrust::reduce_by_key(tempClusterIdBuffer.begin(), tempClusterIdBuffer.end(),
			positions.begin(), clusterMaxIdBuffer.begin(), m_clusterBBMax.begin(),
			thrust::equal_to<int>(), Max<float3>());
	
	thrust::pair<thrust::device_vector<int>::iterator, thrust::device_vector<float3>::iterator> end_min = 
		thrust::reduce_by_key(tempClusterIdBuffer.begin(), tempClusterIdBuffer.end(),
		positions.begin(), clusterMinIdBuffer.begin(), m_clusterBBMin.begin(), 
		thrust::equal_to<int>(), Min<float3>());
	
	// transform the depth values back
	m_numClusters = new_end.second - tileIdBuffer.begin();
	thrust::transform(cluster_k.begin(), cluster_k.begin() + m_numClusters,
		tileIdBuffer.begin(), cluster_k.begin(), add_mul<int>(-maxValue));
	
	// map the cluster-ids to the right pixel position (order was probably destroyed by the sorting step)
	thrust::for_each(
		thrust::make_zip_iterator(thrust::make_tuple(
				pixelIdBuffer.begin(), tempClusterIdBuffer.begin())),
		thrust::make_zip_iterator(thrust::make_tuple(
				pixelIdBuffer.end(), tempClusterIdBuffer.end())),
		Mapping(thrust::raw_pointer_cast(&m_clusterIdBuffer[0])));

	// determine cluster centers
	thrust::device_vector<float3> centers(m_numClusters);
	thrust::transform(m_clusterBBMin.begin(), m_clusterBBMin.begin() + m_numClusters, m_clusterBBMax.begin(), centers.begin(),
		Center<float3>());

	std::vector<float3> tempPos(m_numClusters);
	std::vector<float3> tempMin(m_numClusters);
	std::vector<float3> tempMax(m_numClusters);
	centerPositions.resize(m_numClusters);
	clusterMin.resize(m_numClusters);
	clusterMax.resize(m_numClusters);
	colors.resize(m_numClusters);
	thrust::copy(centers.begin(), centers.end(), tempPos.begin());
	thrust::copy(m_clusterBBMin.begin(), m_clusterBBMin.begin() + m_numClusters, tempMin.begin());
	thrust::copy(m_clusterBBMax.begin(), m_clusterBBMax.begin() + m_numClusters, tempMax.begin());

	for (int i = 0; i < m_numClusters; ++i) {
		centerPositions[i] = make_vec3(tempPos[i]);
		clusterMin[i] = make_vec3(tempMin[i]);
		clusterMax[i] = make_vec3(tempMax[i]);
		colors[i] = glm::vec3(1.f, 1.f, 1.f);
	}
	
	/*
	m_input->positions.resize(width * height);
	m_input->normals.resize(width * height);
		
	m_nodeData.reset(new VisiblePointsBvhNodeData());
	m_nodeData->pixel.resize(width * height);
	VisiblePointsBvhNodeDataParam param;
	param.pixel = thrust::raw_pointer_cast(&m_nodeData->pixel[0]);
	m_nodeData->m_param.reset(new cuda::CudaBuffer<VisiblePointsBvhNodeDataParam>(&param));
	
	// fill vectors with information
	{
		cuda::CudaTimer timer;
		timer.start();
		dim3 dimBlock(32, 32);
		dim3 dimGrid((width  + dimBlock.x - 1) / dimBlock.x,
			(height + dimBlock.y - 1) / dimBlock.y);

		kernel_fillVisiblePoints<<<dimGrid, dimBlock>>>(
			positionSurfaceObject, normalSurfaceObject,
			thrust::raw_pointer_cast(&m_input->positions[0]), 
			thrust::raw_pointer_cast(&m_input->normals[0]), 
			thrust::raw_pointer_cast(&m_nodeData->pixel[0]), 
			width, height);
		timer.stop();
		std::cout << "fill vectors with visible points: " << timer.getTime() << "ms" << std::endl;
	}

	create();
	*/
}


VisiblePointsBvh::~VisiblePointsBvh()
{
}

void VisiblePointsBvh::fillInnerNodes()
{
	dim3 dimBlock(128);
	dim3 dimGrid((m_numLeafs + dimBlock.x - 1) / dimBlock.x);
	kernel_innerNodes<VisiblePointsBvhNodeDataParam><<<dimGrid, dimBlock>>>(
		thrust::raw_pointer_cast(&m_nodes[0]), 
		thrust::raw_pointer_cast(&m_data->parents[0]), 
		thrust::raw_pointer_cast(&m_input->positions[0]), 
		m_data->numLeafs,
		m_nodeData->m_param->getDevicePtr());
}

void VisiblePointsBvh::sort()
{
	thrust::sort_by_key(m_data->morton.begin(), m_data->morton.end(),
		thrust::make_zip_iterator(thrust::make_tuple(
				m_input->positions.begin(), 
				m_input->normals.begin(),
				m_nodeData->pixel.begin()
			)	
		)
	);
}
