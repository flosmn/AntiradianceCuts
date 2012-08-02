#ifndef _C_SIMPLE_KD_TREE_H_
#define _C_SIMPLE_KD_TREE_H_

#include <glm/glm.hpp>

#include "LightTreeTypes.h"

#include "BBox.h"

#include <vector>
#include <unordered_map>

class AVPL;

struct Node
{
	Node(Node* l, Node* r, int i, CLUSTER* c, int d)
		: left(l), right(r), id(i), cluster(c), depth(d) {
		valid = true;
		parent = 0;
		if(left)
			left->parent = this;
		if(right)
			right->parent = this;
		CalcBBox();
	}
	Node() { left = 0; right = 0; parent = 0; cluster = 0; valid = true;}
	
	bool IsLeaf() { return (left == 0 && right == 0);}

	void CalcBBox();
		
	Node* left;
	Node* right;
	Node* parent;
	CLUSTER* cluster;
	BBox bbox;
	int id;
	bool valid;
	int depth;
};

class CSimpleKdTree
{
public:
	CSimpleKdTree();
	~CSimpleKdTree();

	void BuildTree(const std::vector<CLUSTER*>& clusters);
	void Release();

	CLUSTER* GetNearestNeigbour(CLUSTER* query);

	Node* GetHead();

	void Traverse(Node* n);

	void Color(std::vector<AVPL>& avpls, const int cutDepth);

	void MergeClusters(CLUSTER* merged, CLUSTER* c1, CLUSTER* c2);

private:
	Node* BuildTree(
		const std::vector<CLUSTER*>& data_points_sorted_x, 
		const std::vector<CLUSTER*>& data_points_sorted_y,
		const std::vector<CLUSTER*>& data_points_sorted_z,
		int depth);
	
	CLUSTER* GetNearestNeigbour(Node* n, CLUSTER* query);

	void GetAllNodes(Node* n, std::vector<Node*>& leafs);

	CLUSTER* GetNN(CLUSTER* p1, CLUSTER* p2, CLUSTER* query);

	void Color(std::vector<AVPL>& avpls, const int cutDepth, Node* cluster, const int currentDepth, const int colorIndex);

	void Release(Node* n);
		
	glm::vec3 GetRandomColor();
	void InitColors();

	bool SubTreeInvalid(Node* n);
	void UpdateBoundingBoxes(Node* n);

	glm::vec3* m_pColors;
	int m_NumColors;

	Node* m_Head;

	int* m_pLeftIndices;
	int m_LeftIndicesLevel;

	std::unordered_map<CLUSTER*, Node*> m_MapClusterToNode;
};

#endif // _C_SIMPLE_KD_TREE_H