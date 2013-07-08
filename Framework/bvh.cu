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

#define PRINT_DEBUG

std::mt19937 rng;
std::uniform_real_distribution<float> dist01;

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
	if (d==0) return __clz(i ^ j);
	return d;
}

inline __device__ int sign(int i)
{
	return (i < 0) ? -1 : 1;
}

// TODO: shared memory
__global__ void kernel_buildRadixTree(BvhNode* nodes, uint64_t* morton, int* parents, int numNodes)
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
	do {
		if (delta(i, i + (s+t)*d, morton, numNodes) > delta_node) {
			s += t;
		}
		k *= 2;
		t = (l+k-1)/k;
	} while (t > 0);
	const int split = i + s*d + min(d, 0);

	// output child pointers
	BvhNode node;
	node.visited = 0;
	node.left	= min(i, j) == split	? split		: -(split+1);
	node.right	= max(i, j) == split+1	? (split+1)	: -(split+2);

	if (node.left >= 0) {
		parents[node.left] = i;
	} else {
		parents[rootIndex - (node.left+1)] = i;
	}
	if (node.right >= 0) {
		parents[node.right] = i;
	} else {
		parents[rootIndex - (node.right+1)] = i;
	}

	nodes[i] = node;

	if (i == 0) {
		parents[rootIndex] = -1;
	}
}

// TODO: shared memory
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
{

}

Bvh::~Bvh()
{
}

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
	checkTreeIntegrity();

	printDebugRadixTree();
	
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
	
	//// calculate SA of all AABBs
	//float sa = 0;
	//for (int i = 0; i < m_nodes.size(); ++i) {
	//	const BvhNode node = m_nodes[i];
	//	const float3 bbMin = node.bbMin;
	//	const float3 bbMax = node.bbMax;
	//	const float dx = abs(bbMax.x - bbMin.x);
	//	const float dy = abs(bbMax.y - bbMin.y);
	//	const float dz = abs(bbMax.z - bbMin.z);
	//
	//	sa += 0.001f * 2.f * (dx * (dy + dz) + dy * dz); 
	//}
	//std::cout << "sum of aabb surface areas: " << sa << std::endl;
	//// calculate volume of all AABBs
	//float vol = 0;
	//for (int i = 0; i < m_nodes.size(); ++i) {
	//	const BvhNode node = m_nodes[i];
	//	const float3 bbMin = node.bbMin;
	//	const float3 bbMax = node.bbMax;
	//	const float dx = abs(bbMax.x - bbMin.x);
	//	const float dy = abs(bbMax.y - bbMin.y);
	//	const float dz = abs(bbMax.z - bbMin.z);
	//
	//	vol += 0.001f * (dx * dy * dz); 
	//}
	//std::cout << "sum of aabb volumes: " << vol << std::endl;
}

void Bvh::printDebugRadixTree()
{
#ifdef PRINT_DEBUG
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
#endif
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

void Bvh::checkTreeIntegrity()
{
	m_nodesDebug.resize(m_nodes.size());
	thrust::copy(m_nodes.begin(), m_nodes.end(), m_nodesDebug.begin());

	std::vector<int> stack(4096);
	int numNodes = m_nodes.size();
	stack[0] = -1;
	int stack_ptr = 1;
	int max_stack_ptr = 0;
	std::vector<int> checkparent(numNodes);
	std::vector<int> visited(numNodes + m_numLeafs);
	std::fill(checkparent.begin(), checkparent.end(), 0);
	std::fill(visited.begin(), visited.end(), 0);

	while (stack_ptr > 0) {
		int nodeIndex = stack[--stack_ptr];

		if (nodeIndex >= 0) { // leaf
			if (nodeIndex >= m_numLeafs) {
				std::cout << "leaf node index " << nodeIndex << " out of bounds";
			}
			int parent = m_data->parents[nodeIndex];
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
			BvhNode node = m_nodesDebug[-(nodeIndex+1)];
			int idx = m_numLeafs - (nodeIndex+1);
			int parent = m_data->parents[idx];
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
