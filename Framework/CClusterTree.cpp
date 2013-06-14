#include "CClusterTree.h"

#include <iostream>
#include <algorithm>
#include <iterator>

#include "AVPL.h"
#include "Utils\Rand.h"
#include "CTimer.h"
#include "Defines.h"

const int MAX_NUM_AVPLS = 100000;

struct POINT
{
	glm::vec3 position;
	int index;
};

bool SORT_X(const POINT& a1, const POINT& a2);
bool SORT_Y(const POINT& a1, const POINT& a2);
bool SORT_Z(const POINT& a1, const POINT& a2);

CClusterTree::CClusterTree()
{
	m_NumColors = 40;
	m_Head = 0;
	m_ClusterId = 0;	
	m_LeftIndicesLevel = 0;

	m_pClustering = 0;
	m_pLeftIndices = 0;

	m_pClustering = new CLUSTER[2 * MAX_NUM_AVPLS];
	m_pLeftIndices = new int[2 * MAX_NUM_AVPLS];

	memset(m_pClustering, 0, 2 * MAX_NUM_AVPLS * sizeof(CLUSTER));
	memset(m_pLeftIndices, 0, 2 * MAX_NUM_AVPLS * sizeof(int));

	m_indices_sorted_x_0 = new int[MAX_NUM_AVPLS];
	m_indices_sorted_y_0 = new int[MAX_NUM_AVPLS];
	m_indices_sorted_z_0 = new int[MAX_NUM_AVPLS];
	m_indices_sorted_x_1 = new int[MAX_NUM_AVPLS];
	m_indices_sorted_y_1 = new int[MAX_NUM_AVPLS];
	m_indices_sorted_z_1 = new int[MAX_NUM_AVPLS];

	InitColors();
}

CClusterTree::~CClusterTree()
{
	SAFE_DELETE(m_pClustering);
	SAFE_DELETE(m_pLeftIndices);
	SAFE_DELETE(m_pColors);
	SAFE_DELETE(m_indices_sorted_x_0);
	SAFE_DELETE(m_indices_sorted_y_0);
	SAFE_DELETE(m_indices_sorted_z_0);
	SAFE_DELETE(m_indices_sorted_x_1);
	SAFE_DELETE(m_indices_sorted_y_1);
	SAFE_DELETE(m_indices_sorted_z_1);
}
	
void CClusterTree::BuildTree(const std::vector<AVPL>& avpls)
{
	m_ClusterId = 0;		
	m_LeftIndicesLevel = 0;

	m_NumClusters = 2 * (int)avpls.size() - 1;
	
	CTimer timer(CTimer::CPU);
	
	CreateLeafClusters(avpls);
	
	CreateSortedIndices(m_indices_sorted_x_0, m_indices_sorted_y_0, m_indices_sorted_z_0, avpls);
	
	BuildTree(
		m_indices_sorted_x_0, m_indices_sorted_y_0, m_indices_sorted_z_0,
		m_indices_sorted_x_1, m_indices_sorted_y_1, m_indices_sorted_z_1,
		0, (int)avpls.size(), (int)avpls.size(), 0);

	m_Head = &(m_pClustering[m_ClusterId-1]);
	
	SetDepths(m_Head, 0);
}

CLUSTER CClusterTree::BuildTree(int* indices_split_axis_from, int* indices_other_axis_1_from, int* indices_other_axis_2_from, 
		int* indices_split_axis_to, int* indices_other_axis_1_to, int* indices_other_axis_2_to,
		int leftIndex, int rightIndex, int numIndices, int depth)
{
	static int node_Id = 0;

	CLUSTER c;

	if(numIndices == 1)
	{
		// create leaf node
		return m_pClustering[indices_split_axis_from[leftIndex]];
	}
	else if(numIndices == 0)
	{
		std::cout << "subspace already empty" << std::endl;
		return c;
	}
	else
	{
		const int medianIndex = numIndices / 2;
		const int size_left = numIndices / 2;
		const int size_right = numIndices - size_left;
		m_LeftIndicesLevel++;

		for(int i = leftIndex; i < leftIndex + size_left; ++i)
		{
			const int index = indices_split_axis_from[i];
			indices_split_axis_to[i] = index;
			m_pLeftIndices[index] = m_LeftIndicesLevel;
		}

		for(int i = leftIndex + size_left; i < rightIndex; ++i)
		{
			indices_split_axis_to[i] = indices_split_axis_from[i];
		}

		int other_axis_1_left = leftIndex;
		int other_axis_1_right = leftIndex + size_left;
		int other_axis_2_left = leftIndex;
		int other_axis_2_right = leftIndex + size_left;
		for(int i = leftIndex; i < rightIndex; ++i)
		{
			const int index_1 = indices_other_axis_1_from[i];
			if(m_pLeftIndices[index_1] == m_LeftIndicesLevel)
				indices_other_axis_1_to[other_axis_1_left++] = index_1;
			else
				indices_other_axis_1_to[other_axis_1_right++] = index_1;
						
			const int index_2 = indices_other_axis_2_from[i];
			if(m_pLeftIndices[index_2] == m_LeftIndicesLevel)
				indices_other_axis_2_to[other_axis_2_left++] = index_2;
			else
				indices_other_axis_2_to[other_axis_2_right++] = index_2;
		}
		
		// call build recursively
		CLUSTER left = BuildTree(
			indices_other_axis_1_to, indices_other_axis_2_to, indices_split_axis_to,
			indices_other_axis_1_from, indices_other_axis_2_from, indices_split_axis_from, 
			leftIndex, leftIndex + size_left, size_left, depth + 1);
		CLUSTER right = BuildTree(
			indices_other_axis_1_to, indices_other_axis_2_to, indices_split_axis_to,
			indices_other_axis_1_from, indices_other_axis_2_from, indices_split_axis_from,
			leftIndex + size_left, rightIndex, size_right, depth + 1);

		// create inner node
		c = MergeClusters(left, right, depth);
	}

	return c;
}

CLUSTER CClusterTree::MergeClusters(const CLUSTER& l, const CLUSTER& r, int depth)
{
	const float thresh = l.intensity.length() / (l.intensity.length() + r.intensity.length());
	const float p = Rand01();
	
	const float sl = float(l.size);
	const float sr = float(r.size);
	const float norm = 1.f / (sl + sr);
	
	const int id = m_ClusterId;

	if (p <= thresh)
	{
		m_pClustering[id].mean = l.mean;
		m_pClustering[id].normal = l.normal;
		m_pClustering[id].incomingDirection = l.incomingDirection;
		m_pClustering[id].materialIndex = l.materialIndex;
	}
	else
	{
		m_pClustering[id].mean = r.mean;
		m_pClustering[id].normal = r.normal;
		m_pClustering[id].incomingDirection = r.incomingDirection;
		m_pClustering[id].materialIndex = r.materialIndex;
	}
	
	m_pClustering[id].bbox = BBox::Union(l.bbox, r.bbox);
	m_pClustering[id].avplIndex = -1;
	m_pClustering[id].depth = depth;
	m_pClustering[id].left = &(m_pClustering[l.id]);
	m_pClustering[id].right = &(m_pClustering[r.id]);
	m_pClustering[id].intensity = l.intensity + r.intensity;
	m_pClustering[id].size = l.size + r.size;
	m_pClustering[id].id = id;

	m_ClusterId++;
	return m_pClustering[id];
}

void CClusterTree::CreateLeafClusters(const std::vector<AVPL>& avpls)
{
	for(int i = 0; i < avpls.size(); ++i)
	{
		const AVPL avpl = avpls[i];
		const glm::vec3 pos = avpls[i].GetPosition();
		
		const int id = m_ClusterId;
		m_pClustering[id].avplIndex = i;
		m_pClustering[id].id = id;
		m_pClustering[id].bbox = BBox(pos, pos);
		m_pClustering[id].depth = -1;
		m_pClustering[id].intensity = avpls[i].GetIncidentRadiance(); // + avpls[i].GetAntiradiance(avpls[i].GetDirection());
		m_pClustering[id].incomingDirection = avpls[i].GetDirection();
		m_pClustering[id].materialIndex = avpls[i].GetMaterialIndex();
		m_pClustering[id].mean = pos;
		m_pClustering[id].normal = glm::normalize(avpls[i].GetOrientation());
		m_pClustering[id].size = 1;
		m_pClustering[id].left = 0;
		m_pClustering[id].right = 0;

		m_ClusterId++;
	}
}

void CClusterTree::CreateSortedIndices(int* indices_sorted_x, int* indices_sorted_y, int* indices_sorted_z, const std::vector<AVPL>& avpls)
{
	// create sorted vectors
	std::vector<POINT> points;
	for(int i = 0; i < avpls.size(); ++i)
	{
		POINT p;
		p.position = avpls[i].GetPosition();
		p.index = i;
		points.push_back(p);
	}

	std::vector<POINT> points_sorted_x;
	std::vector<POINT> points_sorted_y;
	std::vector<POINT> points_sorted_z;
	
	points_sorted_x.reserve(points.size());
	points_sorted_y.reserve(points.size());
	points_sorted_z.reserve(points.size());
	
	std::copy(points.begin(), points.end(), std::back_inserter(points_sorted_x));
	std::copy(points.begin(), points.end(), std::back_inserter(points_sorted_y));
	std::copy(points.begin(), points.end(), std::back_inserter(points_sorted_z));

	std::sort(points_sorted_x.begin(), points_sorted_x.end(), SORT_X);
	std::sort(points_sorted_y.begin(), points_sorted_y.end(), SORT_Y);
	std::sort(points_sorted_z.begin(), points_sorted_z.end(), SORT_Z);

	for(int i = 0; i < avpls.size(); ++i)
	{
		indices_sorted_x[i] = points_sorted_x[i].index;
		indices_sorted_y[i] = points_sorted_y[i].index;
		indices_sorted_z[i] = points_sorted_z[i].index;
	}

	points.clear();
	points_sorted_x.clear();
	points_sorted_y.clear();
	points_sorted_z.clear();
}

void CClusterTree::Color(std::vector<AVPL>& avpls, const int cutDepth)
{
	Color(avpls, cutDepth, GetHead(), 0, 0);
}

void CClusterTree::Release()
{
	m_Head = 0;
	m_ClusterId = 0;	
	m_LeftIndicesLevel = 0;

	memset(m_pClustering, 0, 2 * MAX_NUM_AVPLS * sizeof(CLUSTER));
	memset(m_pLeftIndices, 0, 2 * MAX_NUM_AVPLS * sizeof(int));
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
		std::vector<CLUSTER> leafs;
		GetAllLeafs(cluster, leafs);
		for (size_t i = 0; i < leafs.size(); ++i)
		{
			avpls[leafs[i].avplIndex].SetColor(m_pColors[colorIndex]);
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

void CClusterTree::GetAllLeafs(CLUSTER* cluster, std::vector<CLUSTER>& leafs)
{
	if(!cluster) return;

	if(cluster->IsLeaf())
	{
		leafs.push_back(*cluster);
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

bool SORT_X(const POINT& p1, const POINT& p2)
{
	return (p1.position.x < p2.position.x);
}

bool SORT_Y(const POINT& p1, const POINT& p2)
{
	return (p1.position.y < p2.position.y);
}

bool SORT_Z(const POINT& p1, const POINT& p2)
{
	return (p1.position.z < p2.position.z);
}