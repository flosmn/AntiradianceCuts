#ifndef _C_CLUSTER_TREE_H_
#define _C_CLUSTER_TREE_H_

#include <glm/glm.hpp>
#include "LightTreeTypes.h"

#include <vector>

class AVPL;
class CTimer;

namespace ClusterTree
{
	class CClusterTree
	{
	public:
		CClusterTree();
		~CClusterTree();
	
		void BuildTree(const std::vector<AVPL*>& avpls);
		void Color(const std::vector<AVPL*>& avpls, const int cutDepth);

		void Release();
		
		CLUSTER* GetHead();
		CLUSTER* GetClustering() const { return m_pClustering; }
		int GetClusteringSize() const { return m_numClusters; }
		
	private:
		CLUSTER* BuildTreeRecurse(int* clusterIds, int numClusters, int depth);
		void CreateLeafClusters(const std::vector<AVPL*>& avpls);
		BBox GetBoundingBox(int* clusterIds, int numClusters);
		CLUSTER* MergeClusters(CLUSTER* leftChild, CLUSTER* rightChild, int depth);

		void Traverse(CLUSTER* cluster);
		void Release(CLUSTER* cluster);
		void Color(const std::vector<AVPL*>& avpls, const int cutDepth, CLUSTER* cluster, const int currentDepth, const int colorIndex);
		void GetAllLeafs(CLUSTER* cluster, std::vector<CLUSTER*>& leafs);
		void SetDepths(CLUSTER* n, int depth);
	
		CLUSTER* m_Head;

		glm::vec3 GetRandomColor();
		void InitColors();

		glm::vec3* m_pColors;
		int m_NumColors;
		int m_ClusterId;
		int m_numClusters;

		CLUSTER* m_pClustering;
	};
}

#endif _C_CLUSTER_TREE_H_