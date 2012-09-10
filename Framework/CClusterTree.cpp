#include "CClusterTree.h"

#include <iostream>
#include <map>
#include <algorithm>
#include <iterator>

#include "AVPL.h"
#include "Utils\Rand.h"

CClusterTree::CClusterTree()
{
	m_NumColors = 40;
	m_Head = 0;
	m_ClusterId = 0;
	m_pClustering = 0;
	m_pLeftIndices = 0;

	InitColors();
}

CClusterTree::~CClusterTree()
{
	delete [] m_pColors;
}
	
void CClusterTree::BuildTree(const std::vector<AVPL>& avpls)
{
	m_ClusterId = 0;

	const int size = (int)avpls.size();
	m_numClusters = 2 * size - 1;

	m_pClustering = new CLUSTER[m_numClusters];		
	m_pLeftIndices = new int[m_numClusters];
	m_LeftIndicesLevel = 0;
	
	std::vector<CLUSTER*> data_points;
	CreateLeafClusters(avpls, data_points);

	// create sorted vectors
	std::vector<CLUSTER*> data_points_sorted_x;
	std::vector<CLUSTER*> data_points_sorted_y;
	std::vector<CLUSTER*> data_points_sorted_z;

	data_points_sorted_x.reserve(data_points.size());
	data_points_sorted_y.reserve(data_points.size());
	data_points_sorted_z.reserve(data_points.size());
	
	std::copy(data_points.begin(), data_points.end(), std::back_inserter(data_points_sorted_x));
	std::copy(data_points.begin(), data_points.end(), std::back_inserter(data_points_sorted_y));
	std::copy(data_points.begin(), data_points.end(), std::back_inserter(data_points_sorted_z));

	std::sort(data_points_sorted_x.begin(), data_points_sorted_x.end(), SORT_X);
	std::sort(data_points_sorted_y.begin(), data_points_sorted_y.end(), SORT_Y);
	std::sort(data_points_sorted_z.begin(), data_points_sorted_z.end(), SORT_Z);
	
	// start to build tree recursively
	m_Head = BuildTree(data_points_sorted_x, data_points_sorted_y, data_points_sorted_z, 0);

	SetDepths(m_Head, 0);
}

CLUSTER* CClusterTree::BuildTree(
		const std::vector<CLUSTER*>& dp_split_axis,
		const std::vector<CLUSTER*>& dp_other_axis_1,
		const std::vector<CLUSTER*>& dp_other_axis_2,
		int depth)
{
	static int node_Id = 0;

	CLUSTER* c;

	if(dp_split_axis.size() == 1)
	{
		// create leaf node
		return dp_split_axis[0];
	}
	else if(dp_split_axis.size() == 0)
	{
		return 0;
	}
	else
	{
		// create inner node
		int numDataPoints = (int)dp_split_axis.size();
		int medianIndex = numDataPoints / 2;
		CLUSTER* median = dp_split_axis[medianIndex];
		
		m_LeftIndicesLevel++;

		// create disjoint sets of points
		std::vector<CLUSTER*> dp_split_axis_left;
		std::vector<CLUSTER*> dp_other_axis_1_left;
		std::vector<CLUSTER*> dp_other_axis_2_left;
		std::vector<CLUSTER*> dp_split_axis_right;
		std::vector<CLUSTER*> dp_other_axis_1_right;
		std::vector<CLUSTER*> dp_other_axis_2_right;

		dp_split_axis_left.reserve(numDataPoints/2 + 1);
		dp_other_axis_1_left.reserve(numDataPoints/2 + 1);
		dp_other_axis_2_left.reserve(numDataPoints/2 + 1);
		dp_split_axis_right.reserve(numDataPoints/2 + 1);
		dp_other_axis_1_right.reserve(numDataPoints/2 + 1);
		dp_other_axis_2_right.reserve(numDataPoints/2 + 1);
				
		for(int i = 0; i < medianIndex; ++i)
		{
			CLUSTER* dp = dp_split_axis[i];
			dp_split_axis_left.push_back(dp);
			m_pLeftIndices[dp->id] = m_LeftIndicesLevel;
		}

		for(int i = medianIndex; i < numDataPoints; ++i)
		{
			CLUSTER* dp = dp_split_axis[i];
			dp_split_axis_right.push_back(dp);
		}

		for(int i = 0; i < numDataPoints; ++i)
		{
			CLUSTER* dp;

			dp = dp_other_axis_1[i];
			if(m_pLeftIndices[dp->id] == m_LeftIndicesLevel)
				dp_other_axis_1_left.push_back(dp);
			else
				dp_other_axis_1_right.push_back(dp);
			
			dp = dp_other_axis_2[i];
			if(m_pLeftIndices[dp->id] == m_LeftIndicesLevel)
				dp_other_axis_2_left.push_back(dp);
			else
				dp_other_axis_2_right.push_back(dp);
		}

		// call build recursively
		CLUSTER* left = BuildTree(
			dp_other_axis_1_left,
			dp_other_axis_2_left,
			dp_split_axis_left,
			depth + 1);
		CLUSTER* right = BuildTree(
			dp_other_axis_1_right,
			dp_other_axis_2_right,
			dp_split_axis_right,
			depth + 1);

		// create inner node
		c = MergeClusters(left, right, depth);
	}

	return c;
}

CLUSTER* CClusterTree::MergeClusters(CLUSTER* l, CLUSTER* r, int depth)
{
	const float sl = float(l->size);
	const float sr = float(r->size);
	const float norm = 1.f / (sl + sr);
	
	CLUSTER* c = &(m_pClustering[m_ClusterId]);

	c->bbox = BBox::Union(l->bbox, r->bbox);
	c->avplIndex = -1;
	c->depth = depth;
	c->left = l;
	c->right = r;
	c->intensity = l->intensity + r->intensity;
	c->mean = norm * (sl * l->mean + sr * r->mean);
	c->normal = norm * (sl * l->normal + sr * r->normal);
	c->size = l->size + r->size;
	c->id = m_ClusterId++;

	return c;
}

void CClusterTree::CreateLeafClusters(const std::vector<AVPL>& avpls,
	std::vector<CLUSTER*>& data_points)
{
	for(int i = 0; i < avpls.size(); ++i)
	{
		AVPL avpl = avpls[i];
		int id = m_ClusterId++;
		CLUSTER* leaf = &(m_pClustering[id]);
		glm::vec3 pos = avpls[i].GetPosition();
		leaf->avplIndex = i;
		leaf->id = id;
		leaf->bbox = BBox(pos, pos);
		leaf->depth = -1;
		leaf->intensity = avpls[i].GetIncidentRadiance() + avpls[i].GetAntiradiance(avpls[i].GetDirection());
		leaf->mean = pos;
		leaf->normal = avpls[i].GetOrientation();
		leaf->size = 1;
		leaf->left = 0;
		leaf->right = 0;
		data_points.push_back(&(m_pClustering[id]));
	}
}

void CClusterTree::Color(std::vector<AVPL>& avpls, const int cutDepth)
{
	Color(avpls, cutDepth, GetHead(), 0, 0);
}

void CClusterTree::Release()
{
	if(m_pClustering)
		delete [] m_pClustering;
	if(m_pLeftIndices)
		delete [] m_pLeftIndices;
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
	
void CClusterTree::Color(std::vector<AVPL>& avpls, const int cutDepth, 
	CLUSTER* cluster, const int currentDepth, const int colorIndex)
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