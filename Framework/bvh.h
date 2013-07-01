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
	thrust::device_vector<uint64_t> morton;
	thrust::device_vector<int> ids;
	thrust::device_vector<int> parents;
};

class BVH
{
public:
	explicit BVH(std::vector<glm::vec3> const& positions,
		std::vector<glm::vec3> const& normals, bool considerNormals);
	~BVH();

	void generateDebugInfo(int level);
	std::vector<glm::vec3>& getColors() { return m_colors; }
	std::vector<glm::vec3>& getBBMins() { return m_bbMins; }
	std::vector<glm::vec3>& getBBMaxs() { return m_bbMaxs; }

private:
	void traverse(Node const& node, int depth, int level);
	void colorChildren(Node const& node, glm::vec3 const& color);
	glm::vec3 getColor();
	void addAABB(float3 const& min, float3 const& max);

	void normalize(thrust::device_vector<float3> const& source,
		thrust::device_vector<float3> const& target);

private:
	BVH_DATA m_data;
	thrust::device_vector<Node> m_nodes;
	std::vector<glm::vec3> m_colors;
	std::vector<glm::vec3> m_bbMins;
	std::vector<glm::vec3> m_bbMaxs;
};

#endif // BVH_H_
