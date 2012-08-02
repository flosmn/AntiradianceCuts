#include "CLightTree.h"

#include "AVPL.h"

#include "CPriorityQueue.h"
#include "CSimpleKdTree.h"
#include "CTimer.h"

#include "Utils\Rand.h"

#include <unordered_set>
#include <iostream>
#include <iterator>

using namespace PriorityQueue;

CLightTree::CLightTree()
{
	m_NumColors = 40;
	m_Head = 0;
	m_ClusteringSize = 0;

	topTime = 0.;
	popTime = 0.;
	findTime = 0.;
	findNNTime = 0.;
	pushTime = 0.;

	findBestCPTimer = new CTimer(CTimer::CPU);

	InitColors();

	m_pPriorityQueue = 0;
	m_pClusters = 0;
	m_pNNAccelerator = new CSimpleKdTree();
}

CLightTree::~CLightTree()
{
	delete m_pNNAccelerator;
	delete m_pPriorityQueue;
}

void CLightTree::BuildTreeNaive(const std::vector<AVPL>& avpls, const float weightNormals)
{
	uint cluster_id = 0;
	m_numToCluster = 0;
	
	if(m_pPriorityQueue)
		delete m_pPriorityQueue;
	m_pPriorityQueue = new PriorityQueue::CPriorityQueue((int)avpls.size());
	
	m_ClusteringSize = 2 * (int)avpls.size() - 1;
	m_pClusters = new CLUSTER[m_ClusteringSize];
	m_pToCluster = new int[m_ClusteringSize];
	memset(m_pToCluster, 0, m_ClusteringSize * sizeof(int));
	
	m_Clustering.reserve(m_ClusteringSize);

	CreateInitialClusters(avpls, &cluster_id);
	
	// no clustering needed
	if(m_numToCluster == 1)
		m_Head = &(m_pClusters[0]);

	while(m_numToCluster > 1)
	{
		while(!m_pPriorityQueue->Empty()) m_pPriorityQueue->DeleteMin();
		
		CreateClusterPairs(m_Clustering, weightNormals, false);
		
		CLUSTER_PAIR best_cp = FindBestClusterPair(weightNormals, false);

		CLUSTER* c1 = best_cp.c1;
		CLUSTER* c2 = best_cp.c2;

		m_Head = MergeClusters(c1, c2, &cluster_id);

		m_pToCluster[best_cp.c1->id] = 0;
		m_pToCluster[best_cp.c2->id] = 0;
		m_pToCluster[m_Head->id] = 1;
		m_numToCluster--;
	}

	SetDepths(GetHead(), 0);
}

void CLightTree::BuildTreeTweakCP(const std::vector<AVPL>& avpls, const float weightNormals)
{
	uint cluster_id = 0;
	m_numToCluster = 0;
	
	if(m_pPriorityQueue)
		delete m_pPriorityQueue;
	m_pPriorityQueue = new PriorityQueue::CPriorityQueue((int)avpls.size());
	
	m_ClusteringSize = 2 * (int)avpls.size() - 1;
	m_pClusters = new CLUSTER[m_ClusteringSize];
	m_pToCluster = new int[m_ClusteringSize];
	memset(m_pToCluster, 0, m_ClusteringSize * sizeof(int));
	
	m_Clustering.reserve(m_ClusteringSize);
	
	CreateInitialClusters(avpls, &cluster_id);
	
	CreateClusterPairs(m_Clustering, weightNormals, false);
	
	// no clustering needed
	if(m_numToCluster == 1)
		m_Head = &(m_pClusters[0]);

	while(m_numToCluster > 1)
	{				
		CLUSTER_PAIR best_cp = FindBestClusterPair(weightNormals, false);
		
		CLUSTER* c1 = best_cp.c1;
		CLUSTER* c2 = best_cp.c2;

		m_Head = MergeClusters(c1, c2, &cluster_id);
		
		m_pToCluster[best_cp.c1->id] = 0;
		m_pToCluster[best_cp.c2->id] = 0;
		m_pToCluster[m_Head->id] = 1;
		m_numToCluster--;
		
		if(m_numToCluster > 1)
		{
			// find the nearest neighbor of the new cluster
			CLUSTER_PAIR cp;
			{
				float dist = 0.f;
				CLUSTER* nn = FindNearestNeighbour(m_Head, m_Clustering, &dist, weightNormals);
				cp.c1 = m_Head;
				cp.c2 = nn;
				cp.dist = dist;
			}
			m_pPriorityQueue->Insert(cp);
		}		
	}

	SetDepths(GetHead(), 0);
}

void CLightTree::BuildTreeTweakNN(const std::vector<AVPL>& avpls, const float weightNormals)
{
	uint cluster_id = 0;
	m_numToCluster = 0;
	
	if(m_pPriorityQueue)
		delete m_pPriorityQueue;
	m_pPriorityQueue = new PriorityQueue::CPriorityQueue((int)avpls.size());
	
	m_ClusteringSize = 2 * (int)avpls.size() - 1;
	m_pClusters = new CLUSTER[m_ClusteringSize];
	m_pToCluster = new int[m_ClusteringSize];
	memset(m_pToCluster, 0, m_ClusteringSize * sizeof(int));
	
	m_Clustering.reserve(m_ClusteringSize);
			
	CTimer timer(CTimer::CPU);
	CTimer timer2(CTimer::CPU);

	timer.Start();
	CreateInitialClusters(avpls, &cluster_id);
	timer.Stop();
	//std::cout << "Create Initial clusters took: " << timer.GetTime() << "ms." << std::endl;

	timer.Start();
	m_pNNAccelerator->BuildTree(m_Clustering);
	timer.Stop();
	//std::cout << "kd-tree creation took: " << timer.GetTime() << "ms." << std::endl;	

	timer.Start();
	CreateClusterPairs(m_Clustering, weightNormals, true);
 	timer.Stop();
	//std::cout << "Cluster pair creation took: " << timer.GetTime() << "ms." << std::endl;
	
	timer.Start();

	// no clustering needed
	if(m_numToCluster == 1)
		m_Head = &(m_pClusters[0]);

	double fbcpTime = 0.;
	double fnnTime = 0.f;
	double mergekdTime = 0.f;
	double mergeTime = 0.f;
	double insertTime = 0.f;
	double deleteinsertTime = 0.f;

	while(m_numToCluster > 1)
	{				
		timer2.Start();
		CLUSTER_PAIR best_cp = FindBestClusterPair(weightNormals, true);
		timer2.Stop();
		fbcpTime += timer2.GetTime();

		CLUSTER* c1 = best_cp.c1;
		CLUSTER* c2 = best_cp.c2;
		
		timer2.Start();
		m_Head = MergeClusters(c1, c2, &cluster_id);
		timer2.Stop();
		mergeTime += timer2.GetTime();

		timer2.Start();
		m_pNNAccelerator->MergeClusters(m_Head, c1, c2);
		timer2.Stop();
		mergekdTime += timer2.GetTime();
		
		timer2.Start();
		m_pToCluster[best_cp.c1->id] = 0;
		m_pToCluster[best_cp.c2->id] = 0;
		m_pToCluster[m_Head->id] = 1;
		m_numToCluster--;
		timer2.Stop();
		deleteinsertTime += timer2.GetTime();

		if(m_numToCluster > 1)
		{
			// find the nearest neighbor of the new cluster
			CLUSTER_PAIR cp;
			{
				float dist = 0.f;
				
				timer2.Start();
				CLUSTER* nn = FindNearestNeighbourWithAccelerator(m_Head, &dist, weightNormals);
				timer2.Stop();
				fnnTime += timer2.GetTime();

				cp.c1 = m_Head;
				cp.c2 = nn;
				cp.dist = dist;
			}
			timer2.Start();
			m_pPriorityQueue->Insert(cp);
			timer2.Stop();
			insertTime += timer2.GetTime();
		}
	}
	timer.Stop();
	/*
	std::cout << "Recursive tree construction took: " << timer.GetTime() << "ms." << std::endl;
	std::cout << "FindBestClusterPair took: " << fbcpTime << "ms." << std::endl;
	std::cout << "FindBestClusterPair: top took: " << topTime << "ms." << std::endl;
	std::cout << "FindBestClusterPair: pop took: " << popTime << "ms." << std::endl;
	std::cout << "FindBestClusterPair: find took: " << findTime << "ms." << std::endl;
	std::cout << "FindBestClusterPair: findNN took: " << findNNTime << "ms." << std::endl;
	std::cout << "FindBestClusterPair: push took: " << pushTime << "ms." << std::endl;
	std::cout << "delete insert time took: " << deleteinsertTime << "ms." << std::endl;
	std::cout << "MergeClusters took: " << mergeTime << "ms." << std::endl;
	std::cout << "MergeClustersKd took: " << mergekdTime << "ms." << std::endl;
	std::cout << "FindNearestNeighbourWithAccelerator took: " << fnnTime << "ms." << std::endl;
	std::cout << "Insert took: " << insertTime << "ms." << std::endl;
	*/
	
	SetDepths(GetHead(), 0);

	m_pNNAccelerator->Release();
}

void CLightTree::CreateInitialClusters(const std::vector<AVPL>& avpls, uint* cluster_id)
{
	for(size_t i = 0; i < avpls.size(); ++i)
	{
		int id = (*cluster_id)++;
		CLUSTER* c = &(m_pClusters[id]);
		
		glm::vec3 pos = avpls[i].GetPosition();
		BBox bbox;
		c->mean = pos;
		c->bbox = BBox(pos, pos);
		c->id = id;
		c->avplIndex = (int)i;
		c->intensity = avpls[i].GetMaxIntensity() + avpls[i].GetMaxAntiintensity();
		c->normal = avpls[i].GetOrientation();
		c->left = 0;
		c->right = 0;
		c->size = 1;

		m_pToCluster[id] = 1;
		m_numToCluster++;
		m_Clustering.push_back(c);
	}
}

void CLightTree::CreateClusterPairs(std::vector<CLUSTER*> clusters, const float weightNormals, bool useAccelerator)
{
	std::vector<CLUSTER*>::iterator clusterIterator1;
	std::vector<CLUSTER*>::iterator clusterIterator2;
		
	int index = 0;
	for(clusterIterator1 = clusters.begin(); clusterIterator1 != clusters.end(); ++ clusterIterator1)
	{
		CLUSTER* c1 = *clusterIterator1;
		
		// search NN
		float dist = 0.f;
		CLUSTER* nn;
		if(useAccelerator) 
			nn = FindNearestNeighbourWithAccelerator(c1, &dist, weightNormals);
		else
			nn = FindNearestNeighbour(c1, clusters, &dist, weightNormals);

		// insert cluster-pair in priority-queue
		CLUSTER_PAIR cp(c1, nn, dist);
		m_pPriorityQueue->Insert(cp);
	}
}

CLUSTER* CLightTree::MergeClusters(CLUSTER* c1, CLUSTER* c2, uint* cluster_id)
{
	const float s1 = float(c1->size);
	const float s2 = float(c2->size);
	const float norm = 1.f / (s1 + s2);
		
	int id = (*cluster_id)++;
	CLUSTER* newCluster = &(m_pClusters[id]);
	newCluster->id = id;
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

CLUSTER* CLightTree::FindNearestNeighbour(CLUSTER* c, std::vector<CLUSTER*> clusters, float* dist, const float weightNormals)
{
	CLUSTER* nn = 0;
	float min_dist = std::numeric_limits<float>::max();

	std::vector<CLUSTER*>::iterator clusterIterator;
	for(clusterIterator = clusters.begin(); clusterIterator != clusters.end(); ++ clusterIterator)
	{
		CLUSTER* cand = *clusterIterator;

		if(*cand == *c || m_pToCluster[cand->id] == 0) continue;

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

CLUSTER_PAIR CLightTree::FindBestClusterPair(const float weightNormals, bool useAccelerator)
{
	CLUSTER_PAIR best_cp;
	float min_dist = std::numeric_limits<float>::max();
	
	std::list<CLUSTER_PAIR> toDelete;
	std::list<CLUSTER_PAIR>::iterator clusterPairIterator;

	bool foundValidCP = false;
	while(!foundValidCP)
	{
		findBestCPTimer->Start();
		CLUSTER_PAIR cp = m_pPriorityQueue->DeleteMin();
		findBestCPTimer->Stop();
		topTime += findBestCPTimer->GetTime();

		// check if the cluster-pair is still valid
		findBestCPTimer->Start();
		bool find = m_pToCluster[cp.c1->id] == 1;
		findBestCPTimer->Stop();
		findTime += findBestCPTimer->GetTime();

		if(m_pToCluster[cp.c1->id] == 0)
		{
			continue;
		}
		else if(m_pToCluster[cp.c2->id] == 0)
		{
			float dist = 0.f;
			CLUSTER* nn;
			findBestCPTimer->Start();
			if(useAccelerator) 
				nn = FindNearestNeighbourWithAccelerator(cp.c1, &dist, weightNormals);
			else
				nn = FindNearestNeighbour(cp.c1, m_Clustering, &dist, weightNormals);
			findBestCPTimer->Stop();
			findNNTime += findBestCPTimer->GetTime();

			CLUSTER_PAIR new_cp(cp.c1, nn, dist);

			if(dist < cp.dist)
			{
				best_cp = new_cp;
				foundValidCP = true;
			}
			else
			{
				findBestCPTimer->Start();
				m_pPriorityQueue->Insert(new_cp);
				findBestCPTimer->Stop();
				pushTime += findBestCPTimer->GetTime();
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

void CLightTree::Color(std::vector<AVPL>& avpls, const int cutDepth)
{
	Color(avpls, cutDepth, GetHead(), 0, 0);
}

void CLightTree::Color(std::vector<AVPL>& avpls, const int cutDepth, CLUSTER* cluster, const int currentDepth, const int colorIndex)
{
	if (currentDepth == cutDepth)
	{
		std::vector<CLUSTER*> leafs;
		GetAllLeafs(cluster, leafs);
		for (size_t i = 0; i < leafs.size(); ++i)
		{
			avpls[leafs[i]->avplIndex].SetColor(m_pColors[colorIndex]);
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

void CLightTree::TraverseIterative(CLUSTER* cluster)
{
	int indicesStack[64];
	int stackPointer = 0;
	
	indicesStack[stackPointer++] = cluster->id; // push root node onto stack

	CLUSTER* currentCluster;
	while(!stackPointer == 0)
	{
		currentCluster = m_Clustering[indicesStack[--stackPointer]];
		if(currentCluster->right != 0)
			indicesStack[stackPointer++] = currentCluster->right->id;	// issue traversal of right subtree
		if(currentCluster->left != 0)
			indicesStack[stackPointer++] = currentCluster->left->id;	// issue traversal of left subtree
		
		// process current node
		if(currentCluster->IsLeaf())
			std::cout << "cluster node " << currentCluster->id << " is leaf with avpl index " << currentCluster->avplIndex << std::endl;
		else
			std::cout << "cluster node " << currentCluster->id << " is inner node with child nodes " << currentCluster->left->id << " and " << currentCluster->right->id << std::endl;
	}
}


void CLightTree::Release()
{
	m_Clustering.clear();
	
	if(m_pPriorityQueue)
		m_pPriorityQueue->Release();

	if(m_pClusters)
		delete [] m_pClusters;
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