#ifndef C_LIGHT_TREE_H_
#define C_LIGHT_TREE_H_

class AVPL;

#include <glm/glm.hpp>

#include "BBox.h"
#include "LightTreeTypes.h"
#include "CPriorityQueue.h"

#include <vector>

class CSimpleKdTree;
class CPriorityQueue;
class CTimer;

typedef unsigned int uint;

class CLightTree
{
public:
	CLightTree();
	~CLightTree();

	void BuildTreeNaive(const std::vector<AVPL*>& avpls, const float weightNormals);
	void BuildTreeTweakCP(const std::vector<AVPL*>& avpls, const float weightNormals);
	void BuildTreeTweakNN(const std::vector<AVPL*>& avpls, const float weightNormals);
	void Color(const std::vector<AVPL*>& avpls, const int cutDepth);

	void Release();

	CLUSTER* GetHead();
	CLUSTER* GetClustering() { return m_pClusters; }
	int GetClusteringSize() { return m_ClusteringSize; }

private:

	void Traverse(CLUSTER* cluster);
	void TraverseIterative(CLUSTER* cluster);
	void Release(CLUSTER* cluster);
	void Color(const std::vector<AVPL*>& avpls, const int cutDepth, CLUSTER* cluster, const int currentDepth, const int colorIndex);
	void GetAllLeafs(CLUSTER* cluster, std::vector<CLUSTER*>& leafs);
	void SetDepths(CLUSTER* n, int depth);

	void CreateInitialClusters(const std::vector<AVPL*>& avpls, uint* cluster_id);
	
	void CreateClusterPairs(std::vector<CLUSTER*> clusters, const float weightNormals, bool useAccelerator);

	CLUSTER* MergeClusters(CLUSTER* c1, CLUSTER* c2, uint* cluster_id);
	CLUSTER* FindNearestNeighbour(CLUSTER* c, std::vector<CLUSTER*> clusters, float* dist, const float weightNormals);
	CLUSTER* FindNearestNeighbourWithAccelerator(CLUSTER* c, float* dist, const float weightNormals);

	CLUSTER_PAIR FindBestClusterPair(const float weightNormals, bool useAccelerator);
	
	CLUSTER* m_Head;

	glm::vec3 GetRandomColor();
	void InitColors();

	glm::vec3* m_pColors;
	int m_NumColors;

	std::vector<CLUSTER*> m_Clustering;

	CLUSTER* m_pClusters;

	CSimpleKdTree* m_pNNAccelerator;
	PriorityQueue::CPriorityQueue* m_pPriorityQueue;

	double topTime;
	double popTime;
	double findTime;
	double findNNTime;
	double pushTime;
	CTimer* findBestCPTimer;

	int* m_pToCluster;
	int m_numToCluster;
	int m_ClusteringSize;

};

#endif C_LIGHT_TREE_H_