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
	thrust::device_vector<int> parents;
	int numLeafs;
	int numNodes;
};

struct BvhInput 
{
	thrust::device_vector<float3> positions;
	thrust::device_vector<float3> normals;
};

struct BvhParam
{
	int numLeafs;
	int numNodes;
	BvhNode* nodes;
	float3* positions;
	float3* normals;
};

class Bvh
{
public:
	explicit Bvh(bool considerNormals);
	virtual ~Bvh();
	
	virtual void fillInnerNodes() = 0;
	virtual void sort() = 0;
	void create();

	void generateDebugInfo(int level);
	std::vector<glm::vec3>& getColors() { return m_colors; }
	std::vector<glm::vec3>& getPositions() { return m_positions; }
	std::vector<glm::vec3>& getBBMins() { return m_bbMins; }
	std::vector<glm::vec3>& getBBMaxs() { return m_bbMaxs; }

	BvhParam* getBvhParam() { return m_param->getDevicePtr(); }	

private:
	void traverse(BvhNode const& node, int depth, int level);
	void colorChildren(BvhNode const& node, glm::vec3 const& color);
	glm::vec3 getColor();
	void addAABB(float3 const& min, float3 const& max);

	void normalize(thrust::device_vector<float3> const& source,
		thrust::device_vector<float3> const& target);
	void printDebugRadixTree();

protected:
	std::unique_ptr<BvhInput> m_input;
	std::unique_ptr<BvhData> m_data;
	thrust::device_vector<BvhNode> m_nodes;
	int m_numLeafs;
	bool m_considerNormals;

private:
	// for debug output
	std::vector<glm::vec3> m_colors;
	std::vector<glm::vec3> m_positions;
	std::vector<glm::vec3> m_bbMins;
	std::vector<glm::vec3> m_bbMaxs;

	std::unique_ptr<cuda::CudaBuffer<BvhParam>> m_param;
};

struct AvplBvhNodeDataParam
{
	int* size;
	int* materialIndex;
	float* randomNumbers;
	float3* position;
	float3* normal;
	float3* incRadiance;
	float3* incDirection;
};

struct AvplBvhNodeData
{
	AvplBvhNodeData(std::vector<AVPL> const& avpls);

	thrust::device_vector<int> size;
	thrust::device_vector<int> materialIndex;
	thrust::device_vector<float> randomNumbers;
	thrust::device_vector<float3> position;
	thrust::device_vector<float3> normal;
	thrust::device_vector<float3> incRadiance;
	thrust::device_vector<float3> incDirection;
	std::unique_ptr<cuda::CudaBuffer<AvplBvhNodeDataParam>> m_param; 
};

class AvplBvh : public Bvh
{
public:
	AvplBvh(std::vector<AVPL> const& avpls, bool considerNormals);

	~AvplBvh();
	
	virtual void fillInnerNodes();
	virtual void sort();

	void testTraverse();

	AvplBvhNodeDataParam* getAvplBvhNodeDataParam() { return m_nodeData->m_param->getDevicePtr(); }

private:
	std::unique_ptr<AvplBvhNodeData> m_nodeData;

	//data for inner nodes
};

#endif // BVH_H_
