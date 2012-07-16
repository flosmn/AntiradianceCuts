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
	std::vector<CLUSTER*> GetClustering() { return m_Clustering; }

private:

	void Traverse(CLUSTER* cluster);
	void TraverseIterative(CLUSTER* cluster);
	void Release(CLUSTER* cluster);
	void Color(const std::vector<AVPL*>& avpls, const int cutDepth, CLUSTER* cluster, const int currentDepth, const int colorIndex);
	void GetAllLeafs(CLUSTER* cluster, std::vector<CLUSTER*>& leafs);
	void SetDepths(CLUSTER* n, int depth);

	void CreateInitialClusters(const std::vector<AVPL*>& avpls, std::unordered_set<CLUSTER*, HASH_CLUSTER, EQ_CLUSTER>& toCluster, uint* cluster_id);
	
	void CreateClusterPairs(
		const std::unordered_set<CLUSTER*, HASH_CLUSTER, EQ_CLUSTER>& toCluster,
		const float weightNormals, bool useAccelerator);

	CLUSTER* MergeClusters(CLUSTER* c1, CLUSTER* c2, uint* cluster_id);
	CLUSTER* FindNearestNeighbour(CLUSTER* c, float* dist, const std::unordered_set<CLUSTER*, HASH_CLUSTER, EQ_CLUSTER>& toCluster, const float weightNormals);
	CLUSTER* FindNearestNeighbourWithAccelerator(CLUSTER* c, float* dist, const float weightNormals);

	CLUSTER_PAIR FindBestClusterPair(
		const std::unordered_set<CLUSTER*, HASH_CLUSTER, EQ_CLUSTER>& toCluster,
		const float weightNormals, bool useAccelerator);
	
	CLUSTER* m_Head;

	glm::vec3 GetRandomColor();
	void InitColors();

	glm::vec3* m_pColors;
	int m_NumColors;

	std::vector<CLUSTER*> m_Clustering;

	CSimpleKdTree* m_pNNAccelerator;
	PriorityQueue::CPriorityQueue* m_pPriorityQueue;

	double topTime;
	double popTime;
	double findTime;
	double findNNTime;
	double pushTime;
	CTimer* findBestCPTimer;
};

#endif C_LIGHT_TREE_H_