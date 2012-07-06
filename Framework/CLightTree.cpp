#include "CLightTree.h"

#include "AVPL.h"

#include "Utils\Rand.h"

#include <unordered_set>
#include <iostream>

struct CLUSTER
{
	CLUSTER() {
		left = 0;
		right = 0;
	}

	uint id;
	int avplIndex;

	CLUSTER* left;
	CLUSTER* right;

	uint size;
	BBox bbox;
	glm::vec3 mean;
	glm::vec3 intensity;
	glm::vec3 normal;
	
	bool operator<(const CLUSTER c) const {
		return (id < c.id);
	}

	bool operator==(const CLUSTER c) const {
		return (id == c.id);
	}

	bool IsLeaf() const {
		return (left == 0 && right == 0);
	}
};

struct CLUSTER_PAIR
{
	CLUSTER* c1;
	CLUSTER* c2;

	bool operator==(const CLUSTER_PAIR cp) const {
		return (c1->id == cp.c1->id && c2->id == cp.c2->id);
	}

	float dist;
};

struct HASH_CLUSTER
{
    size_t operator()(const CLUSTER* c) const
    {
        std::hash<int> H;
        return H(c->id);
    }
};

struct EQ_CLUSTER
{
    bool operator()(const CLUSTER* c1, const CLUSTER* c2) const {
        return c1->id==c2->id;
    }
};

float dist(const CLUSTER& c1, const CLUSTER& c2, const float weightNormals)
{
	//const float I = glm::length(float(c1.size) * c1.intensity + float(c2.size) * c2.intensity) / float(float(c1.size) + float(c2.size));
	const BBox bbox = BBox::Union(c1.bbox, c2.bbox);
	const float A = glm::length(bbox.pMax - bbox.pMin);
	//const float B = glm::dot(c1.normal, c2.normal);

	return A; //I * ( A * A + weightNormals * weightNormals * (1 - B) * (1 - B));
}

CLightTree::CLightTree()
{
	m_NumColors = 20;

	InitColors();
}

CLightTree::~CLightTree()
{
}

void CLightTree::BuildTree(const std::vector<AVPL*>& avpls, const float weightNormals)
{
	uint cluster_id = 0;

	std::list<CLUSTER_PAIR> clusterPairs;
	std::list<CLUSTER_PAIR>::iterator clusterPairIterator;
	
	std::unordered_set<CLUSTER*, HASH_CLUSTER, EQ_CLUSTER> toCluster; 
	std::unordered_set<CLUSTER*, HASH_CLUSTER, EQ_CLUSTER>::iterator clusterIterator1;
	std::unordered_set<CLUSTER*, HASH_CLUSTER, EQ_CLUSTER>::iterator clusterIterator2;

	for(size_t i = 0; i < avpls.size(); ++i)
	{
		CLUSTER* c = new CLUSTER();

		BBox bbox;
		c->mean = bbox.pMin = bbox.pMax = avpls[i]->GetPosition();
		c->bbox = bbox;
		c->id = cluster_id++;
		c->avplIndex = (int)i;
		c->intensity; avpls[i]->GetIntensity(avpls[i]->GetOrientation());
		c->normal = avpls[i]->GetOrientation();
		c->left = 0;
		c->right = 0;
		c->size = 1;

		toCluster.insert(c);
	}
	
	while(toCluster.size() > 1)
	{
		clusterPairs.clear();
		
		for(clusterIterator1 = toCluster.begin(); clusterIterator1 != toCluster.end(); ++ clusterIterator1)
		{
			CLUSTER* c1 = *clusterIterator1;
			
			// search NN
			CLUSTER* nn;
			float min_dist = std::numeric_limits<float>::max();
			for(clusterIterator2 = toCluster.begin(); clusterIterator2 != toCluster.end(); ++ clusterIterator2)
			{
				CLUSTER* c2 = *clusterIterator2;
				
				if(*c1 == *c2) continue;
				
				float dist_temp = dist(*c1, *c2, weightNormals);
				if(dist_temp <= min_dist)
				{
					min_dist = dist_temp;
					nn = c2;
				}
			}

			// insert cluster-pair in priority-queue
			CLUSTER_PAIR cp;
			cp.c1 = c1;
			cp.c2 = nn;
			cp.dist = min_dist;
			clusterPairs.insert(clusterPairs.end(), cp);
		}
		
		CLUSTER_PAIR best_cp;
		float min_dist = std::numeric_limits<float>::max();
		for(clusterPairIterator = clusterPairs.begin(); clusterPairIterator != clusterPairs.end(); ++clusterPairIterator)
		{
			CLUSTER_PAIR cp = *clusterPairIterator;
			if(cp.dist <= min_dist)
			{
				min_dist = cp.dist;
				best_cp = cp;
			}
		}

		CLUSTER* c1 = best_cp.c1;
		CLUSTER* c2 = best_cp.c2;

		const float s1 = float(c1->size);
		const float s2 = float(c1->size);
		const float norm = 1.f / (s1 + s2);
		
		m_Head = new CLUSTER();
		m_Head->id = cluster_id++;
		m_Head->avplIndex = -1;
		m_Head->intensity = norm * (s1 * c1->intensity + s2 * c2->intensity);
		m_Head->mean = norm * (s1 * c1->mean + s2 * c2->mean);
		m_Head->normal = norm * (s1 * c1->normal + s2 * c2->normal);
		m_Head->size = uint(s1 + s2);
		m_Head->left = c1;
		m_Head->right = c2;
		m_Head->bbox = BBox::Union(c1->bbox, c2->bbox);

		toCluster.erase(best_cp.c1);
		toCluster.erase(best_cp.c2);
		toCluster.insert(m_Head);
	}
}

void CLightTree::BuildTreeTweakCP(const std::vector<AVPL*>& avpls, const float weightNormals)
{
	uint cluster_id = 0;

	std::list<CLUSTER_PAIR> clusterPairs;
	std::list<CLUSTER_PAIR>::iterator clusterPairIterator;
	
	std::unordered_set<CLUSTER*, HASH_CLUSTER, EQ_CLUSTER> toCluster; 
	std::unordered_set<CLUSTER*, HASH_CLUSTER, EQ_CLUSTER>::iterator clusterIterator1;
	std::unordered_set<CLUSTER*, HASH_CLUSTER, EQ_CLUSTER>::iterator clusterIterator2;

	for(size_t i = 0; i < avpls.size(); ++i)
	{
		CLUSTER* c = new CLUSTER();

		BBox bbox;
		c->mean = bbox.pMin = bbox.pMax = avpls[i]->GetPosition();
		c->bbox = bbox;
		c->id = cluster_id++;
		c->avplIndex = (int)i;
		c->intensity; avpls[i]->GetIntensity(avpls[i]->GetOrientation());
		c->normal = avpls[i]->GetOrientation();
		c->left = 0;
		c->right = 0;
		c->size = 1;

		toCluster.insert(c);
	}
	
	for(clusterIterator1 = toCluster.begin(); clusterIterator1 != toCluster.end(); ++ clusterIterator1)
	{
		CLUSTER* c1 = *clusterIterator1;
		
		// search NN
		CLUSTER* nn;
		float min_dist = std::numeric_limits<float>::max();
		for(clusterIterator2 = toCluster.begin(); clusterIterator2 != toCluster.end(); ++ clusterIterator2)
		{
			CLUSTER* c2 = *clusterIterator2;
			
			if(*c1 == *c2) continue;
			
			float dist_temp = dist(*c1, *c2, weightNormals);
			if(dist_temp <= min_dist)
			{
				min_dist = dist_temp;
				nn = c2;
			}
		}

		// insert cluster-pair in priority-queue
		CLUSTER_PAIR cp;
		cp.c1 = c1;
		cp.c2 = nn;
		cp.dist = min_dist;
		clusterPairs.insert(clusterPairs.end(), cp);
	}

	while(toCluster.size() > 1)
	{				
		CLUSTER_PAIR best_cp;
		{
			float min_dist = std::numeric_limits<float>::max();
			std::list<CLUSTER_PAIR> toDelete;
			for(clusterPairIterator = clusterPairs.begin(); clusterPairIterator != clusterPairs.end(); ++clusterPairIterator)
			{
				CLUSTER_PAIR cp = *clusterPairIterator;
				
				// check if the cluster-pair is still valid
				if(toCluster.find(cp.c1) == toCluster.cend() || toCluster.find(cp.c2) == toCluster.cend())
				{
					toDelete.insert(toDelete.begin(), cp);
					continue;
				}

				if(cp.dist <= min_dist)
				{
					min_dist = cp.dist;
					best_cp = cp;
				}
			}
			for(clusterPairIterator = toDelete.begin(); clusterPairIterator != toDelete.end(); ++clusterPairIterator)
			{
				CLUSTER_PAIR cp = *clusterPairIterator;
				clusterPairs.remove(cp);
			}
			toDelete.clear();
		}
		CLUSTER* c1 = best_cp.c1;
		CLUSTER* c2 = best_cp.c2;

		const float s1 = float(c1->size);
		const float s2 = float(c1->size);
		const float norm = 1.f / (s1 + s2);
		
		m_Head = new CLUSTER();
		m_Head->id = cluster_id++;
		m_Head->avplIndex = -1;
		m_Head->intensity = norm * (s1 * c1->intensity + s2 * c2->intensity);
		m_Head->mean = norm * (s1 * c1->mean + s2 * c2->mean);
		m_Head->normal = norm * (s1 * c1->normal + s2 * c2->normal);
		m_Head->size = uint(s1 + s2);
		m_Head->left = c1;
		m_Head->right = c2;
		m_Head->bbox = BBox::Union(c1->bbox, c2->bbox);

		toCluster.erase(best_cp.c1);
		toCluster.erase(best_cp.c2);
		toCluster.insert(m_Head);

		if(toCluster.size() > 1)
		{
			// find the nearest neighbor of the new cluster
			CLUSTER_PAIR cp;
			{
				CLUSTER* nn = 0;
				float min_dist = std::numeric_limits<float>::max();
				for(clusterIterator1 = toCluster.begin(); clusterIterator1 != toCluster.end(); ++ clusterIterator1)
				{
					CLUSTER* c = *clusterIterator1;

					if(*c == *m_Head) continue;

					const float temp_dist = dist(*m_Head, *c, weightNormals);
					if(temp_dist <= min_dist)
					{
						min_dist = temp_dist;
						nn = c;
					}
				}
				cp.c1 = m_Head;
				cp.c2 = nn;
				cp.dist = min_dist;
			}
			clusterPairs.insert(clusterPairs.end(), cp);
		}
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

void CLightTree::Release()
{
	Release(GetHead());
}

void CLightTree::Release(CLUSTER* cluster)
{
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