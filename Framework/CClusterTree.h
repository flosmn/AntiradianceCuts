#ifndef _C_CLUSTER_TREE_H_
#define _C_CLUSTER_TREE_H_

#include <glm/glm.hpp>
#include "LightTreeTypes.h"

#include <vector>

class AVPL;
class CTimer;

class CClusterTree
{
public:
	CClusterTree();
	~CClusterTree();

	void BuildTree(const std::vector<AVPL>& avpls);
	void Color(std::vector<AVPL>& avpls, const int cutDepth);

	void Release();
	
	CLUSTER* GetHead();
	CLUSTER* GetClustering() const { return m_pClustering; }
	int GetClusteringSize() const { return m_NumClusters; }
	void Traverse(CLUSTER* cluster);
	
private:
	CLUSTER BuildTree(
		int* indices_split_axis_from, int* indices_other_axis_1_from, int* indices_other_axis_2_from, 
		int* indices_split_axis_to, int* indices_other_axis_1_to, int* indices_other_axis_2_to,
		int leftIndex, int rightIndex, int numIndices, int depth);
	
	void CreateSortedIndices(int* indices_sorted_x, int* indices_sorted_y, int* indices_sorted_z, const std::vector<AVPL>& avpls);

	void CreateLeafClusters(const std::vector<AVPL>& avpls);

	BBox GetBoundingBox(int* clusterIds, int numClusters);
	CLUSTER MergeClusters(const CLUSTER& leftChild, const CLUSTER& rightChild, int depth);
	void CreateInnerClusters();
	
	void Color(std::vector<AVPL>& avpls, const int cutDepth, CLUSTER* cluster, const int currentDepth, const int colorIndex);
	void GetAllLeafs(CLUSTER* cluster, std::vector<CLUSTER>& leafs);
	void SetDepths(CLUSTER* n, int depth);

	CLUSTER* m_Head;

	glm::vec3 GetRandomColor();
	void InitColors();

	glm::vec3* m_pColors;
	int m_NumColors;
	int m_ClusterId;
	int m_NumClusters;

	CLUSTER* m_pClustering;

	int* m_pLeftIndices;
	int m_LeftIndicesLevel;

	int * m_indices_sorted_x_0;
	int * m_indices_sorted_y_0;
	int * m_indices_sorted_z_0;
	int * m_indices_sorted_x_1;
	int * m_indices_sorted_y_1;
	int * m_indices_sorted_z_1;
};


#endif _C_CLUSTER_TREE_H_