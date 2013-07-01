#ifndef BVH_H_
#define BVH_H_

#include <thrust/device_vector.h>
#include <cuda_runtime.h>

#include "CudaResources/cudaUtil.hpp"

#include <glm/glm.hpp>

#include <vector>
#include <iostream>

// negative numbers indicate pointers to inner nodes
// positive numbers indicate pointers to leaf nodes
struct BvhNode
{
	int left;
	int right;
	float3 bbMin;
	float3 bbMax;
	int visited;
};

class BvhNodeData
{
public:
	__device__ __host__ void cluster(const int target, const int left, const int right) { }
};

struct BvhData
{
	thrust::device_vector<uint64_t> morton;
	thrust::device_vector<int> ids;
	thrust::device_vector<int> parents;
	int numLeafs;
	int numNodes;
};

struct BvhInput 
{
	thrust::device_vector<float3> positions;
	thrust::device_vector<float3> normals;
};

class Bvh
{
public:
	explicit Bvh(BvhInput* bvhInput, BvhNodeData* nodeData, bool considerNormals);
	~Bvh();

	void generateDebugInfo(int level);
	std::vector<glm::vec3>& getColors() { return m_colors; }
	std::vector<glm::vec3>& getBBMins() { return m_bbMins; }
	std::vector<glm::vec3>& getBBMaxs() { return m_bbMaxs; }

private:
	void traverse(BvhNode const& node, int depth, int level);
	void colorChildren(BvhNode const& node, glm::vec3 const& color);
	glm::vec3 getColor();
	void addAABB(float3 const& min, float3 const& max);

	void normalize(thrust::device_vector<float3> const& source,
		thrust::device_vector<float3> const& target);

private:
	BvhInput* m_input;
	BvhNodeData* m_nodeData;
	std::unique_ptr<BvhData> m_data;
	thrust::device_vector<BvhNode> m_nodes;
	std::vector<glm::vec3> m_colors;
	std::vector<glm::vec3> m_bbMins;
	std::vector<glm::vec3> m_bbMaxs;
	
	std::vector<float3> m_positionsDebug;
	std::vector<int> m_idsDebug;
	std::vector<BvhNode> m_nodesDebug;
};

class AvplBvh
{
public:
	AvplBvh(std::vector<glm::vec3> const& positions,
		std::vector<glm::vec3> const& normals, bool considerNormals);

	~AvplBvh();

	void generateDebugInfo(int level) { m_bvh->generateDebugInfo(level); }
	std::vector<glm::vec3>& getColors() { return m_bvh->getColors(); }
	std::vector<glm::vec3>& getBBMins() { return m_bvh->getBBMins(); }
	std::vector<glm::vec3>& getBBMaxs() { return m_bvh->getBBMaxs(); }

private:
	std::unique_ptr<BvhInput> m_input;
	std::unique_ptr<BvhNodeData> m_nodeData;
	std::unique_ptr<Bvh> m_bvh;
};

#endif // BVH_H_
