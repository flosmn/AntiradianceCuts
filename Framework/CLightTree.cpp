#include "CLightTree.h"

#include "AVPL.h"

#include <unordered_set>

struct CLUSTER
{
	uint id;

	CLUSTER* left;
	CLUSTER* right;

	uint size;
	BBox bbox;
	glm::vec3 mean;
	glm::vec3 intensity;
	glm::vec3 normal;

	bool operator<(const CLUSTER c1) const {
		return (id < c1.id);
	}

	bool operator==(const CLUSTER c1) const {
		return (id == c1.id);
	}
};

struct CLUSTER_PAIR
{
	CLUSTER* c1;
	CLUSTER* c2;

	float dist;
};

struct HASH_CLUSTER
{
    size_t operator()(const CLUSTER c) const
    {
        std::hash<int> H;
        return H(c.id);
    }
};

struct EQ_CLUSTER
{
    bool operator()(const CLUSTER c1, const CLUSTER c2) const {
        return c1.id==c2.id;
    }
};

float dist(const CLUSTER& c1, const CLUSTER& c2, const float& weightNormals)
{
	//const float I = glm::length(float(c1.size) * c1.intensity + float(c2.size) * c2.intensity) / float(float(c1.size) + float(c2.size));
	const BBox bbox = BBox::Union(c1.bbox, c2.bbox);
	const float A = glm::length(bbox.pMax - bbox.pMin);
	//const float B = glm::dot(c1.normal, c2.normal);

	return A; //I * ( A * A + weightNormals * weightNormals * (1 - B) * (1 - B));
}

CLightTree::CLightTree()
{
}

CLightTree::~CLightTree()
{
}

void CLightTree::BuildTree(const std::vector<AVPL*>& avpls, const float& weightNormals)
{
	uint cluster_id = 0;

	std::list<CLUSTER_PAIR> clusterPairs;
	std::list<CLUSTER_PAIR>::iterator clusterPairIterator;
	
	std::unordered_set<CLUSTER, HASH_CLUSTER, EQ_CLUSTER> toCluster; 
	std::unordered_set<CLUSTER, HASH_CLUSTER, EQ_CLUSTER>::iterator clusterIterator1;
	std::unordered_set<CLUSTER, HASH_CLUSTER, EQ_CLUSTER>::iterator clusterIterator2;

	for(size_t i = 0; i < avpls.size(); ++i)
	{
		CLUSTER c;

		BBox bbox;
		c.mean = bbox.pMin = bbox.pMax = avpls[i]->GetPosition();
		c.bbox = bbox;
		c.id = cluster_id++;
		c.intensity; avpls[i]->GetIntensity(avpls[i]->GetOrientation());
		c.normal = avpls[i]->GetOrientation();
		c.left = 0;
		c.right = 0;
		c.size = 1;

		toCluster.insert(c);
	}

	for(clusterIterator1 = toCluster.begin(); clusterIterator1 != toCluster.end(); ++ clusterIterator1)
	{
		CLUSTER c1 = *clusterIterator1;
		
		// search NN
		CLUSTER nn;
		float min_dist = std::numeric_limits<float>::max();
		for(clusterIterator2 = toCluster.begin(); clusterIterator2 != toCluster.end(); ++ clusterIterator2)
		{
			CLUSTER c2 = *clusterIterator2;
			
			if(c1 == c2) continue;
			
			float dist_temp = dist(c1, c2, weightNormals);
			if(dist_temp <= min_dist)
			{
				min_dist = dist_temp;
				nn = c2;
			}
		}

		// insert cluster-pair in priority-queue
		CLUSTER_PAIR cp;
		cp.c1 = &c1;
		cp.c2 = &nn;
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

	toCluster.erase(*(best_cp.c1));
	toCluster.erase(*(best_cp.c2));

}