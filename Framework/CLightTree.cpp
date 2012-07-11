#include "CLightTree.h"

#include "AVPL.h"

#include "CSimpleKdTree.h"
#include "CTimer.h"

#include "Utils\Rand.h"

#include <unordered_set>
#include <iostream>
#include <iterator>

CLightTree::CLightTree()
{
	m_NumColors = 40;
	m_Head = 0;

	InitColors();

	m_pNNAccelerator = new CSimpleKdTree();
}

CLightTree::~CLightTree()
{
	delete m_pNNAccelerator;
}

void CLightTree::BuildTreeNaive(const std::vector<AVPL*>& avpls, const float weightNormals)
{
	uint cluster_id = 0;
	
	std::priority_queue<CLUSTER_PAIR, std::vector<CLUSTER_PAIR>, std::greater<std::vector<CLUSTER_PAIR>::value_type>> clusterPairs;
	std::unordered_set<CLUSTER*, HASH_CLUSTER, EQ_CLUSTER> toCluster; 

	CreateInitialClusters(avpls, toCluster, &cluster_id);
	
	// no clustering needed
	if(toCluster.size() == 1)
		m_Head = *(toCluster.begin());

	while(toCluster.size() > 1)
	{
		while(!clusterPairs.empty()) clusterPairs.pop();
		
		CreateClusterPairs(toCluster, clusterPairs, weightNormals, false);
		
		CLUSTER_PAIR best_cp = FindBestClusterPair(toCluster, clusterPairs, weightNormals, false);

		CLUSTER* c1 = best_cp.c1;
		CLUSTER* c2 = best_cp.c2;

		m_Head = MergeClusters(c1, c2, &cluster_id);

		toCluster.erase(best_cp.c1);
		toCluster.erase(best_cp.c2);
		toCluster.insert(m_Head);
	}

	SetDepths(GetHead(), 0);
}

void CLightTree::BuildTreeTweakCP(const std::vector<AVPL*>& avpls, const float weightNormals)
{
	uint cluster_id = 0;

	std::priority_queue<CLUSTER_PAIR, std::vector<CLUSTER_PAIR>, std::greater<std::vector<CLUSTER_PAIR>::value_type>> clusterPairs;
	std::unordered_set<CLUSTER*, HASH_CLUSTER, EQ_CLUSTER> toCluster;
	
	CreateInitialClusters(avpls, toCluster, &cluster_id);

	std::vector<CLUSTER*> cluster;
	std::copy(toCluster.begin(), toCluster.end(), back_inserter(cluster));
	
	CreateClusterPairs(toCluster, clusterPairs, weightNormals, false);
	
	// no clustering needed
	if(toCluster.size() == 1)
		m_Head = *(toCluster.begin());

	while(toCluster.size() > 1)
	{				
		CLUSTER_PAIR best_cp = FindBestClusterPair(toCluster, clusterPairs, weightNormals, false);
		
		CLUSTER* c1 = best_cp.c1;
		CLUSTER* c2 = best_cp.c2;

		m_Head = MergeClusters(c1, c2, &cluster_id);
		
		toCluster.erase(best_cp.c1);
		toCluster.erase(best_cp.c2);
		toCluster.insert(m_Head);
		
		if(toCluster.size() > 1)
		{
			// find the nearest neighbor of the new cluster
			CLUSTER_PAIR cp;
			{
				float dist = 0.f;
				CLUSTER* nn = FindNearestNeighbour(m_Head, &dist, toCluster, weightNormals);
				cp.c1 = m_Head;
				cp.c2 = nn;
				cp.dist = dist;
			}
			clusterPairs.push(cp);
		}		
	}

	SetDepths(GetHead(), 0);
}

void CLightTree::BuildTreeTweakNN(const std::vector<AVPL*>& avpls, const float weightNormals)
{
	uint cluster_id = 0;

	std::priority_queue<CLUSTER_PAIR, std::vector<CLUSTER_PAIR>, std::greater<std::vector<CLUSTER_PAIR>::value_type>> clusterPairs;
	std::unordered_set<CLUSTER*, HASH_CLUSTER, EQ_CLUSTER> toCluster;
	
	CTimer timer(CTimer::CPU);

	timer.Start();
	CreateInitialClusters(avpls, toCluster, &cluster_id);
	timer.Stop();
	//std::cout << "Create Initial clusters took: " << timer.GetTime() << "ms." << std::endl;

	std::vector<CLUSTER*> cluster;
	std::copy(toCluster.begin(), toCluster.end(), back_inserter(cluster));
	
	timer.Start();
	m_pNNAccelerator->BuildTree(cluster);
	timer.Stop();
	//std::cout << "Accelerator creation took: " << timer.GetTime() << "ms." << std::endl;	

	timer.Start();
	CreateClusterPairs(toCluster, clusterPairs, weightNormals, true);
	timer.Stop();
	//std::cout << "Cluster pair creation took: " << timer.GetTime() << "ms." << std::endl;
	
	timer.Start();

	// no clustering needed
	if(toCluster.size() == 1)
		m_Head = *(toCluster.begin());

	while(toCluster.size() > 1)
	{				
		CLUSTER_PAIR best_cp = FindBestClusterPair(toCluster, clusterPairs, weightNormals, true);
		
		CLUSTER* c1 = best_cp.c1;
		CLUSTER* c2 = best_cp.c2;

		m_Head = MergeClusters(c1, c2, &cluster_id);
		m_pNNAccelerator->MergeClusters(m_Head, c1, c2);

		toCluster.erase(best_cp.c1);
		toCluster.erase(best_cp.c2);
		toCluster.insert(m_Head);
		
		if(toCluster.size() > 1)
		{
			// find the nearest neighbor of the new cluster
			CLUSTER_PAIR cp;
			{
				float dist = 0.f;
				CLUSTER* nn = FindNearestNeighbourWithAccelerator(m_Head, &dist, weightNormals);
				cp.c1 = m_Head;
				cp.c2 = nn;
				cp.dist = dist;
			}
			clusterPairs.push(cp);
		}
	}
	timer.Stop();
	//std::cout << "Recursive tree construction took: " << timer.GetTime() << "ms." << std::endl;
	
	SetDepths(GetHead(), 0);

	m_pNNAccelerator->Release();
}

void CLightTree::CreateInitialClusters(const std::vector<AVPL*>& avpls, std::unordered_set<CLUSTER*, HASH_CLUSTER, EQ_CLUSTER>& toCluster, uint* cluster_id)
{
	for(size_t i = 0; i < avpls.size(); ++i)
	{
		CLUSTER* c = new CLUSTER();

		BBox bbox;
		c->mean = bbox.pMin = bbox.pMax = avpls[i]->GetPosition();
		c->bbox = bbox;
		c->id = (*cluster_id)++;
		c->avplIndex = (int)i;
		c->intensity = avpls[i]->GetIntensity(avpls[i]->GetOrientation());
		c->normal = avpls[i]->GetOrientation();
		c->left = 0;
		c->right = 0;
		c->size = 1;

		toCluster.insert(c);
		m_Clustering.push_back(c);
	}
}

void CLightTree::CreateClusterPairs(const std::unordered_set<CLUSTER*, HASH_CLUSTER, EQ_CLUSTER>& toCluster, 
	std::priority_queue<CLUSTER_PAIR, std::vector<CLUSTER_PAIR>, std::greater<std::vector<CLUSTER_PAIR>::value_type>>& clusterPairs,
	const float weightNormals, bool useAccelerator)
{
	std::unordered_set<CLUSTER*, HASH_CLUSTER, EQ_CLUSTER>::iterator clusterIterator1;
	std::unordered_set<CLUSTER*, HASH_CLUSTER, EQ_CLUSTER>::iterator clusterIterator2;
	
	for(clusterIterator1 = toCluster.begin(); clusterIterator1 != toCluster.end(); ++ clusterIterator1)
	{
		CLUSTER* c1 = *clusterIterator1;
		
		// search NN
		float dist = 0.f;
		CLUSTER* nn;
		if(useAccelerator) 
			nn = FindNearestNeighbourWithAccelerator(c1, &dist, weightNormals);
		else
			nn = FindNearestNeighbour(c1, &dist, toCluster, weightNormals);

		// insert cluster-pair in priority-queue
		CLUSTER_PAIR cp(c1, nn, dist);
		clusterPairs.push(cp);
	}
}

CLUSTER* CLightTree::MergeClusters(CLUSTER* c1, CLUSTER* c2, uint* cluster_id)
{
	const float s1 = float(c1->size);
	const float s2 = float(c2->size);
	const float norm = 1.f / (s1 + s2);
		
	CLUSTER* newCluster = new CLUSTER();
	newCluster->id = (*cluster_id)++;
	newCluster->avplIndex = -1;
	newCluster->intensity = c1->intensity + c2->intensity;
	newCluster->mean = norm * (s1 * c1->mean + s2 * c2->mean);
	newCluster->normal = norm * (s1 * c1->normal + s2 * c2->normal);
	newCluster->size = uint(s1 + s2);
	newCluster->left = c1;
	newCluster->right = c2;
	newCluster->bbox = BBox::Union(c1->bbox, c2->bbox);
	
	m_Clustering.push_back(newCluster);

	return newCluster;
}

CLUSTER* CLightTree::FindNearestNeighbour(CLUSTER* c, float* dist, const std::unordered_set<CLUSTER*, HASH_CLUSTER, EQ_CLUSTER>& toCluster, const float weightNormals)
{
	CLUSTER* nn = 0;
	float min_dist = std::numeric_limits<float>::max();

	std::unordered_set<CLUSTER*, HASH_CLUSTER, EQ_CLUSTER>::iterator clusterIterator;
	for(clusterIterator = toCluster.begin(); clusterIterator != toCluster.end(); ++ clusterIterator)
	{
		CLUSTER* cand = *clusterIterator;

		if(*cand == *c) continue;

		const float temp_dist = c->Distance(cand, weightNormals);
		if(temp_dist <= min_dist)
		{
			min_dist = temp_dist;
			nn = cand;
		}		
	}

	*dist = min_dist;

	return nn;
}

CLUSTER* CLightTree::FindNearestNeighbourWithAccelerator(CLUSTER* c, float* dist, const float weightNormals)
{
	CLUSTER* nn = 0;
	float min_dist = std::numeric_limits<float>::max();

	nn = m_pNNAccelerator->GetNearestNeigbour(c);
	
	*dist = c->Distance(nn, weightNormals);

	return nn;
}

CLUSTER_PAIR CLightTree::FindBestClusterPair(const std::unordered_set<CLUSTER*, HASH_CLUSTER, EQ_CLUSTER>& toCluster,
	std::priority_queue<CLUSTER_PAIR, std::vector<CLUSTER_PAIR>, std::greater<std::vector<CLUSTER_PAIR>::value_type>>& clusterPairs,
	const float weightNormals, bool useAccelerator)
{
	CLUSTER_PAIR best_cp;
	float min_dist = std::numeric_limits<float>::max();
	
	std::list<CLUSTER_PAIR> toDelete;
	std::list<CLUSTER_PAIR>::iterator clusterPairIterator;

	bool foundValidCP = false;
	while(!foundValidCP)
	{
		CLUSTER_PAIR cp = clusterPairs.top();
		clusterPairs.pop();
			
		// check if the cluster-pair is still valid
		if(toCluster.find(cp.c1) == toCluster.cend())
		{
			continue;
		}		
		else if(toCluster.find(cp.c2) == toCluster.cend())
		{
			float dist = 0.f;
			CLUSTER* nn;
			if(useAccelerator) 
				nn = FindNearestNeighbourWithAccelerator(cp.c1, &dist, weightNormals);
			else
				nn = FindNearestNeighbour(cp.c1, &dist, toCluster, weightNormals);
			
			CLUSTER_PAIR new_cp(cp.c1, nn, dist);

			if(dist < cp.dist)
			{
				best_cp = new_cp;
				foundValidCP = true;
			}
			else
			{
				clusterPairs.push(new_cp);
			}
		}
		else
		{
			best_cp = cp;
			foundValidCP = true;
		}
	}	

	return best_cp;
}

void CLightTree::Color(const std::vector<AVPL*>& avpls, const int cutDepth)
{
	Color(avpls, cutDepth, GetHead(), 0, 0);
}

void CLightTree::Color(const std::vector<AVPL*>& avpls, const int cutDepth, CLUSTER* cluster, const int currentDepth, const int colorIndex)
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

void CLightTree::Traverse(CLUSTER* cluster)
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

void CLightTree::Release()
{
	m_Clustering.clear();

	Release(GetHead());
}

void CLightTree::Release(CLUSTER* cluster)
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

void CLightTree::GetAllLeafs(CLUSTER* cluster, std::vector<CLUSTER*>& leafs)
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

void CLightTree::SetDepths(CLUSTER* n, int depth)
{
	if(!n) return;

	n->depth = depth;

	if(!n->IsLeaf())
	{
		SetDepths(n->left, depth + 1);
		SetDepths(n->right, depth + 1);
	}
}

CLUSTER* CLightTree::GetHead()
{
	return m_Head;
}

glm::vec3 CLightTree::GetRandomColor()
 {
	 glm::vec3 color = glm::vec3(Rand01(), Rand01(), Rand01());
	 return color;
 }

void CLightTree::InitColors()
{
	m_pColors = new glm::vec3[m_NumColors];

	for(int i = 0; i < m_NumColors; ++i)
	{
		m_pColors[i] = GetRandomColor();
	}
}