#include "CClusterTree.h"

#include <iostream>
#include <map>
#include <algorithm>

#include "AVPL.h"
#include "CTimer.h"
#include "Utils\Rand.h"

using namespace ClusterTree;

CClusterTree::CClusterTree()
{
	m_NumColors = 40;
	m_Head = 0;
	m_ClusterId = 0;
	m_pClustering = 0;

	InitColors();
}

CClusterTree::~CClusterTree()
{
	delete [] m_pColors;
}
	
void CClusterTree::BuildTree(const std::vector<AVPL*>& avpls)
{
	m_ClusterId = 0;

	const int size = (int)avpls.size();
	m_numClusters = 2 * size - 1;

	m_pClustering = new CLUSTER[m_numClusters];
	int* leafIds = new int[size];
	for(int i = 0; i < size; ++i)
		leafIds[i] = i;

	CreateLeafClusters(avpls);
	
	m_Head = BuildTreeRecurse(leafIds, size, 0);

	SetDepths(m_Head, 0);

	delete [] leafIds;
}

CLUSTER* CClusterTree::BuildTreeRecurse(int* clusterIds, int numClusters, int depth)
{
	if(numClusters == 0)
		return 0;

	if(numClusters == 1)
		return &(m_pClustering[clusterIds[0]]);
	
	BBox bbox = GetBoundingBox(clusterIds, numClusters);

	int axis = bbox.MaximumExtent();
	float spatialMedian = 0.5f * (bbox.pMax[axis] + bbox.pMin[axis]);

	int* clusterIdsLeft = new int[numClusters];
	int* clusterIdsRight = new int[numClusters];
	int numClustersLeft = 0;
	int numClustersRight = 0;
	
	for(int i = 0; i < numClusters; ++i)
	{
		CLUSTER c = m_pClustering[clusterIds[i]];
		if(c.mean[axis] < spatialMedian)
			clusterIdsLeft[numClustersLeft++] = c.id;
		else
			clusterIdsRight[numClustersRight++] = c.id;
	}

	CLUSTER* leftChild = BuildTreeRecurse(clusterIdsLeft, numClustersLeft, depth + 1);
	CLUSTER* rightChild = BuildTreeRecurse(clusterIdsRight, numClustersRight, depth + 1);

	delete [] clusterIdsLeft;
	delete [] clusterIdsRight;

	return MergeClusters(leftChild, rightChild, depth);
}

CLUSTER* CClusterTree::MergeClusters(CLUSTER* l, CLUSTER* r, int depth)
{
	CLUSTER* c = &(m_pClustering[m_ClusterId]);

	c->bbox = BBox::Union(l->bbox, r->bbox);
	c->avplIndex = -1;
	c->depth = depth;
	c->left = l;
	c->right = r;
	c->intensity = l->intensity + r->intensity;
	c->mean = 0.5f * (l->mean + r->mean);
	c->normal = 0.5f * (l->normal + r->normal);
	c->size = l->size + r->size;
	c->id = m_ClusterId++;

	return c;
}

void CClusterTree::CreateLeafClusters(const std::vector<AVPL*>& avpls)
{
	for(int i = 0; i < avpls.size(); ++i)
	{
		AVPL* avpl = avpls[i];
		int id = m_ClusterId++;
		CLUSTER* leaf = &(m_pClustering[id]);
		glm::vec3 pos = avpl->GetPosition();
		leaf->avplIndex = i;
		leaf->id = id;
		leaf->bbox = BBox(pos, pos);
		leaf->depth = -1;
		leaf->intensity = avpl->GetMaxIntensity() + avpl->GetMaxAntiintensity();
		leaf->mean = pos;
		leaf->normal = avpl->GetOrientation();
		leaf->size = 1;
		leaf->left = 0;
		leaf->right = 0;
	}
}

BBox CClusterTree::GetBoundingBox(int* clusterIds, int numClusters)
{
	if(numClusters == 0)
		return BBox(glm::vec3(std::numeric_limits<float>::min()), glm::vec3(std::numeric_limits<float>::min()));

	BBox bbox = m_pClustering[clusterIds[0]].bbox;
	for(int i = 1; i < numClusters; ++i)
	{
		bbox = BBox::Union(bbox,  m_pClustering[clusterIds[i]].bbox);
	}

	return bbox;
}

void CClusterTree::Color(const std::vector<AVPL*>& avpls, const int cutDepth)
{
	Color(avpls, cutDepth, GetHead(), 0, 0);
}

void CClusterTree::Release()
{
	if(m_pClustering)
		delete [] m_pClustering;
}

void CClusterTree::Traverse(CLUSTER* cluster)
{
	if(cluster->IsLeaf())
	{
		std::cout << "cluster node " << cluster->id << " is leaf with avpl index " << cluster->avplIndex << std::endl;
	}
	else
	{
		std::cout << "cluster node " << cluster->id << " is inner node with child nodes " << cluster->left->id << " and " << cluster->right->id << std::endl;
	
		Traverse(cluster->left);
		Traverse(cluster->right);
	}
}
	
void CClusterTree::Color(const std::vector<AVPL*>& avpls, const int cutDepth, 
	CLUSTER* cluster, const int currentDepth, const int colorIndex)
{
	if (currentDepth == cutDepth)
	{
		std::vector<CLUSTER*> leafs;
		GetAllLeafs(cluster, leafs);
		for (size_t i = 0; i < leafs.size(); ++i)
		{
			avpls[leafs[i]->avplIndex]->SetColor(m_pColors[colorIndex]);
		}
		leafs.clear();
	}
	else
	{
		if (cluster->IsLeaf()) return;

		Color(avpls, cutDepth, cluster->left,  currentDepth + 1, int(Rand01() * (m_NumColors - 1)) );
		Color(avpls, cutDepth, cluster->right, currentDepth + 1, int(Rand01() * (m_NumColors - 1)) );
	}
}		

void CClusterTree::Release(CLUSTER* cluster)
{
	if(!cluster) return;

	if(cluster->IsLeaf())
	{
		delete cluster;
		cluster = 0;
	}
	else
	{
		Release(cluster->left);
		Release(cluster->right);
	}
}

void CClusterTree::GetAllLeafs(CLUSTER* cluster, std::vector<CLUSTER*>& leafs)
{
	if(!cluster) return;

	if(cluster->IsLeaf())
	{
		leafs.push_back(cluster);
	}
	else
	{
		GetAllLeafs(cluster->left, leafs);
		GetAllLeafs(cluster->right, leafs);
	}
}

void CClusterTree::SetDepths(CLUSTER* n, int depth)
{
	if(!n) return;

	n->depth = depth;

	if(!n->IsLeaf())
	{
		SetDepths(n->left, depth + 1);
		SetDepths(n->right, depth + 1);
	}
}

CLUSTER* CClusterTree::GetHead()
{
	return m_Head;
}

glm::vec3 CClusterTree::GetRandomColor()
 {
	 glm::vec3 color = glm::vec3(Rand01(), Rand01(), Rand01());
	 return color;
 }

void CClusterTree::InitColors()
{
	m_pColors = new glm::vec3[m_NumColors];

	for(int i = 0; i < m_NumColors; ++i)
	{
		m_pColors[i] = GetRandomColor();
	}
}