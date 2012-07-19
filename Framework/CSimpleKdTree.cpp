#include "CSimpleKdTree.h"

#include "Utils\Rand.h"
#include "AVPL.h"
#include "CTimer.h"

#include <algorithm>
#include <unordered_set>
#include <iostream>
#include <iterator>

bool SORT_X(CLUSTER* p1, CLUSTER* p2)
{
	return (p1->mean.x < p2->mean.x);
}

bool SORT_Y(CLUSTER* p1, CLUSTER* p2)
{
	return (p1->mean.y < p2->mean.y);
}

bool SORT_Z(CLUSTER* p1, CLUSTER* p2)
{
	return (p1->mean.z < p2->mean.z);
}

float Distance(CLUSTER* p1, CLUSTER* p2)
{
	if(p1 == 0 || p2 == 0)
		return std::numeric_limits<float>::max();

	return glm::length(p1->mean - p2->mean);
}

CSimpleKdTree::CSimpleKdTree()
{
	m_NumColors = 40;
	InitColors();
}

CSimpleKdTree::~CSimpleKdTree()
{
	if(m_pColors)
	{
		delete [] m_pColors;
		m_pColors = 0;
	}
}

void CSimpleKdTree::BuildTree(const std::vector<CLUSTER*>& data_points)
{
	m_pLeftIndices = new int[data_points.size()];
	m_LeftIndicesLevel = 0;
	m_MapClusterToNode = std::unordered_map<CLUSTER*, Node*>(2 * data_points.size());
	
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
}

Node* CSimpleKdTree::BuildTree(
		const std::vector<CLUSTER*>& dp_split_axis,
		const std::vector<CLUSTER*>& dp_other_axis_1,
		const std::vector<CLUSTER*>& dp_other_axis_2,
		int depth)
{
	static int node_Id = 0;

	Node* n;

	if(dp_split_axis.size() == 1)
	{
		// create leaf node
		CLUSTER* c = dp_split_axis[0];
		n = new Node(0, 0, node_Id++, c, depth);
		m_MapClusterToNode[c] = n;
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

		for(int i = medianIndex + 1; i < numDataPoints; ++i)
		{
			CLUSTER* dp = dp_split_axis[i];
			dp_split_axis_right.push_back(dp);
		}

		for(int i = 0; i < numDataPoints; ++i)
		{
			CLUSTER* dp;
			dp = dp_other_axis_1[i];
			if(dp != median)
			{
				if(m_pLeftIndices[dp->id] == m_LeftIndicesLevel)
					dp_other_axis_1_left.push_back(dp);
				else
					dp_other_axis_1_right.push_back(dp);
			}

			dp = dp_other_axis_2[i];
			if(dp != median)
			{
				if(m_pLeftIndices[dp->id] == m_LeftIndicesLevel)
					dp_other_axis_2_left.push_back(dp);
				else
					dp_other_axis_2_right.push_back(dp);
			}
		}

		// call build recursively
		Node* left = BuildTree(
			dp_other_axis_1_left,
			dp_other_axis_2_left,
			dp_split_axis_left,
			depth + 1);
		Node* right = BuildTree(
			dp_other_axis_1_right,
			dp_other_axis_2_right,
			dp_split_axis_right,
			depth + 1);

		// create inner node
		n = new Node(left, right, node_Id++, median, depth);
		m_MapClusterToNode[median] = n;
	}

	return n;
}

CLUSTER* CSimpleKdTree::GetNearestNeigbour(CLUSTER* query)
{
	return GetNearestNeigbour(GetHead(), query);
}

CLUSTER* CSimpleKdTree::GetNearestNeigbour(Node* n, CLUSTER* query)
{
	CLUSTER* nn = 0;
	
	if(!n) return nn;

	if(n->IsLeaf() && n->valid && n->cluster != query)
	{
		nn = n->cluster;
	}
	else if(!n->IsLeaf())
	{
		float distLeft = std::numeric_limits<float>::max();
		float distRight = std::numeric_limits<float>::max();

		if(n->left)
			distLeft = Distance(query, n->left->cluster);
		if(n->right)
			distRight = Distance(query, n->right->cluster);

		CLUSTER* best_cand;
		if(distLeft <= distRight)
		{
			best_cand = GetNearestNeigbour(n->left, query);
			
			// check for NN in right subtree
			if(n->right && n->right->bbox.Distance(query->mean) <= Distance(best_cand, query))
			//if(!best_cand || (n->right && query->UpperBound(best_cand).Intersects(n->right->cluster->bbox)))
			{
				best_cand = GetNN(best_cand, GetNearestNeigbour(n->right, query), query);
			}
		}
		else
		{
			best_cand = GetNearestNeigbour(n->right, query);
			
			// check for NN in right subtree
			if(n->left && n->left->bbox.Distance(query->mean) <= Distance(best_cand, query))
			//if(!best_cand || (n->left && query->UpperBound(best_cand).Intersects(n->left->cluster->bbox)))
			{
				best_cand = GetNN(best_cand, GetNearestNeigbour(n->left, query), query);
			}
		}

		// check if root of subtree is better NN
		if(n->valid)
		{
			best_cand = GetNN(best_cand, n->cluster, query);
		}
		
		nn = best_cand;
	}

	return nn;
}

CLUSTER* CSimpleKdTree::GetNN(CLUSTER* p1, CLUSTER* p2, CLUSTER* query)
{
	if(p1 == 0 || p1 == query)
		return p2;
	if(p2 == 0 || p2 == query)
		return p1;

	if(Distance(p1, query) < Distance(p2, query)) 
		return p1;

	return p2;
}

void CSimpleKdTree::Release()
{
	m_MapClusterToNode.clear();

	Release(GetHead());	
}

void CSimpleKdTree::Release(Node* n)
{
	if(!n) return;
	
	if(n->IsLeaf())
	{
		delete n;
		n = 0;
	}
	else
	{
		Release(n->left);
		Release(n->right);
	}
}

void CSimpleKdTree::Traverse(Node* n)
{
	if(!n) return;

	if(n->IsLeaf())
	{
		std::cout << "Node " << n->id << " is leaf. DP index: " <<n->cluster->avplIndex << std::endl;
	}
	else
	{
		std::cout << "Node " << n->cluster->avplIndex << " is inner node with children "
			<< (n->left ? n->left->id : -1) << " and " << (n->right ? n->right->id : -1) << std::endl;
		
		Traverse(n->left);
		Traverse(n->right);
	}
}

void CSimpleKdTree::MergeClusters(CLUSTER* merged, CLUSTER* c1, CLUSTER* c2)
{
	int d1 = m_MapClusterToNode[c1]->depth;
	int d2 = m_MapClusterToNode[c2]->depth;
	
	CLUSTER* lower;
	CLUSTER* higher;
	
	if(d1 > d2)
	{
		lower = c1;
		higher = c2;
	}
	else
	{
		lower = c2;
		higher = c1;
	}

	Node* nHigher = m_MapClusterToNode[higher];
	nHigher->cluster = merged;
	m_MapClusterToNode[merged] = nHigher;
	
	Node* nLower = m_MapClusterToNode[lower];
	nLower->valid = false;

	if(nLower->parent)
		UpdateBoundingBoxes(nLower->parent);

	if(SubTreeInvalid(nLower))
	{
		if(nLower->parent)
		{
			if(nLower->parent->left == nLower)
				nLower->parent->left = 0;
			else
				nLower->parent->right = 0;
		}
		Release(nLower);
	}
	
	m_MapClusterToNode[lower] = 0;
	m_MapClusterToNode[higher] = 0;
}

Node* CSimpleKdTree::GetHead()
{
	return m_Head;
}

void CSimpleKdTree::Color(const std::vector<AVPL*>& avpls, const int cutDepth)
{
	Color(avpls, cutDepth, GetHead(), 0, 0);
}

void CSimpleKdTree::Color(const std::vector<AVPL*>& avpls, const int cutDepth, Node* n, const int currentDepth, const int colorIndex)
{
	if (currentDepth == cutDepth)
	{
		std::vector<Node*> leafs;
		GetAllNodes(n, leafs);
		for (size_t i = 0; i < leafs.size(); ++i)
		{
			avpls[leafs[i]->cluster->avplIndex]->SetColor(m_pColors[colorIndex]);
		}
		leafs.clear();
	}
	else
	{
		if (n->IsLeaf()) return;

		Color(avpls, cutDepth, n->left,  currentDepth + 1, int(Rand01() * (m_NumColors - 1)) );
		Color(avpls, cutDepth, n->right, currentDepth + 1, int(Rand01() * (m_NumColors - 1)) );
	}
}

void CSimpleKdTree::GetAllNodes(Node* n, std::vector<Node*>& nodes)
{
	if(!n) return;

	nodes.push_back(n);

	if(!n->IsLeaf())
	{
		GetAllNodes(n->left, nodes);
		GetAllNodes(n->right, nodes);
	}
}

bool CSimpleKdTree::SubTreeInvalid(Node* n)
{
	if(!n) return true;

	if(n->valid)
		return false;

	if(!n->IsLeaf())
	{
		return SubTreeInvalid(n->left) && SubTreeInvalid(n->right);
	}

	return true;
}

void CSimpleKdTree::UpdateBoundingBoxes(Node* n)
{
	if(!n) return;

	n->CalcBBox();

	UpdateBoundingBoxes(n->parent);
}

glm::vec3 CSimpleKdTree::GetRandomColor()
 {
	 glm::vec3 color = glm::vec3(Rand01(), Rand01(), Rand01());
	 return color;
 }

void CSimpleKdTree::InitColors()
{
	m_pColors = new glm::vec3[m_NumColors];

	for(int i = 0; i < m_NumColors; ++i)
	{
		m_pColors[i] = GetRandomColor();
	}
}

void Node::CalcBBox()
{
	BBox self = BBox(cluster->mean, cluster->mean);
	if(left == 0 && right == 0)	bbox = self;
	else if(left == 0)			bbox = BBox::Union(self, right->bbox);
	else if(right == 0)			bbox = BBox::Union(self, left->bbox);
	else						bbox = BBox::Union(BBox::Union(left->bbox, right->bbox), self);
}