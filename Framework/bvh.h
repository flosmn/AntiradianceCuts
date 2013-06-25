#ifndef BVH_H_
#define BVH_H_

#include <thrust/device_vector.h>

#include <glm/glm.hpp>

#include <vector>
#include <iostream>

// negative numbers indicate pointers to inner nodes
// positive numbers indicate pointers to leaf nodes
struct Node
{
	int left;
	int right;
	float3 bbMin;
	float3 bbMax;
	int visited;
};

struct BVH_DATA
{
	thrust::device_vector<float3> positions;
	thrust::device_vector<int> morton;
	thrust::device_vector<int> ids;
	thrust::device_vector<int> parents;
};

class BVH
{
public:
	explicit BVH(std::vector<glm::vec3> const& positions);
	~BVH();

private:
	BVH_DATA m_data;
	thrust::device_vector<Node> m_nodes;
};

#endif // BVH_H_
