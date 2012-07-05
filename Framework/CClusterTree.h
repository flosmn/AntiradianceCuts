#ifndef _C_CLUSTER_TREE_H_
#define _C_CLUSTER_TREE_H_

#include <glm/glm.hpp>

#include <vector>

class AVPL;

struct DATA_POINT
{
	int key;
	glm::vec3 position;
	glm::vec3 normal;
};

struct Node
{
	Node* leftChild;
	Node* rightChild;
	int kvp_index; // key-value-pair index
	int id;
};

class CClusterTree
{
public:
	CClusterTree();
	~CClusterTree();

	void BuildTree(std::vector<AVPL*>& avpls);
	void Traverse(Node* node);
	void Release();
	void Release(Node* node);
	
	Node* GetHead();

	void ColorAVPLs(std::vector<AVPL*>& avpls, int depth);

private:
	void BuildTree(DATA_POINT* data_points, int num_data_points);

	float SIM(const DATA_POINT& p1, const DATA_POINT& p2);

	void ColorNodes(std::vector<AVPL*>& avpls, int level, Node* node, int depth, int colorIndex);
	void GetAllLeafs(Node* node, std::vector<Node*>& leafs);

	void InitColors();

	bool IsLeaf(Node* node);

	glm::vec3 GetRandomColor();

	Node* m_Head;

	int m_NumDataPoints;
	DATA_POINT* m_pDataPoints;

	int m_NumColors;
	glm::vec3* m_pColors;
};

#endif _C_CLUSTER_TREE_H_