#ifndef BVH_H_
#define BVH_H_

#include <thrust/device_vector.h>
#include <cuda_runtime.h>

#include "CudaResources/cudaUtil.hpp"
#include "CudaResources/cudaBuffer.hpp"

#include "AVPL.h"

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
	explicit Bvh(bool considerNormals);
	virtual ~Bvh();
	
	virtual void fillInnerNodes() = 0;
	void create();

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

protected:
	std::unique_ptr<BvhInput> m_input;
	std::unique_ptr<BvhData> m_data;
	thrust::device_vector<BvhNode> m_nodes;
	int m_numLeafs;
	bool m_considerNormals;

private:
	// for debug output
	std::vector<glm::vec3> m_colors;
	std::vector<glm::vec3> m_bbMins;
	std::vector<glm::vec3> m_bbMaxs;
};

struct AvplBvhNodeData;

class AvplBvh : public Bvh
{
public:
	AvplBvh(std::vector<AVPL> const& avpls, bool considerNormals);

	~AvplBvh();
	
	virtual void fillInnerNodes();

private:
	std::unique_ptr<AvplBvhNodeData> m_nodeData;

	//data for inner nodes
};

#endif // BVH_H_
