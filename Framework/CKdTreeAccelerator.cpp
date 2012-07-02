#include "CKdTreeAccelerator.h"

#include "CPrimitive.h"

#include <algorithm>
#include <limits>
#include <iostream>

static const float INFINITY = std::numeric_limits<float>::max();
static const int MAX_TO_TRAVERSE = 64;

struct KdAccelNode;

CKdTreeAccelerator::CKdTreeAccelerator(const std::vector<CPrimitive*>& primitives,
	int intersectionCost, int traversalCost, int maxNumPrimitives, int maxDepth)
	: m_IntersectionCost(intersectionCost), m_TraversalCost(traversalCost),
	m_MaxNumPrimitives(maxNumPrimitives), m_MaxDepth(maxDepth)

{
	m_Primitives = primitives;

	m_NextFreeNode = 0;
	m_NumAllocatedNodes = 0;
	m_EmptyBonus = 0.f;

	if(m_MaxDepth <= 0)
		m_MaxDepth = int(8.f + 1.3f * glm::log2(float(m_Primitives.size())));
}

CKdTreeAccelerator::~CKdTreeAccelerator()
{
}

void CKdTreeAccelerator::BuildTree()
{
	// compute bounds for kd-tree construction
	std::vector<BBox> primitiveBounds;
	primitiveBounds.reserve(m_Primitives.size());
	for(uint i = 0; i < m_Primitives.size(); ++i)
	{
		CPrimitive* prim = m_Primitives[i];
		BBox b = prim->GetBBox();
		m_BoundingBox = BBox::Union(m_BoundingBox, b);
		primitiveBounds.push_back(b);
	}

	// allocate memory for construction
	BoundEdge* edges[3];
	for(int i = 0; i < 3; ++i)
		edges[i] = new BoundEdge[2 * m_Primitives.size()];
	uint* prims0 = new uint[m_Primitives.size()];
	uint* prims1 = new uint[(m_MaxDepth+1) * m_Primitives.size()];

	// initialize overlapping primitives (all at the beginning)
	uint* overlappingPrimitives = new uint[m_Primitives.size()];
	for(uint i = 0; i < m_Primitives.size(); ++i)
	{
		overlappingPrimitives[i] = i;
	}

	// start to recursively build the tree
	BuildTreeRecursive(0, m_BoundingBox, primitiveBounds, overlappingPrimitives,
		m_Primitives.size(), m_MaxDepth, edges, prims0, prims1, 0);

	// release allocated memory
	for(int i = 0; i < 3; ++i)
		delete [] edges[i];
	delete [] prims0;
	delete [] prims1;
}

void CKdTreeAccelerator::BuildTreeRecursive(int node, const BBox& nodeBounds,
		const std::vector<BBox>& primitiveBounds, uint* overlappingPrimitives,
		int numOverlappingPrimitives, int depth, BoundEdge* edges[3],
		uint* prims0, uint* prims1, int badRefines)
{
	// Get next free node from nodes array
	if(m_NextFreeNode == m_NumAllocatedNodes)
	{
		int newNumAllocatedNodes = std::max(2 * m_NumAllocatedNodes, 512);
		KdAccelNode* n = new KdAccelNode[newNumAllocatedNodes];
		if(m_NumAllocatedNodes > 0)
		{
			memcpy(n, m_Nodes, m_NumAllocatedNodes * sizeof(KdAccelNode));
			delete [] m_Nodes;
		}
		m_Nodes = n;
		m_NumAllocatedNodes = newNumAllocatedNodes;
	}
	++m_NextFreeNode;

	// Initialize leaf node of termination criteria is met
	if(numOverlappingPrimitives <= m_MaxNumPrimitives || depth == 0)
	{
		m_Nodes[node].InitLeaf(overlappingPrimitives, numOverlappingPrimitives);
	}

	// Initialize interior node and continue recursion
	InitializeInteriorNode(node, nodeBounds, primitiveBounds, 
		overlappingPrimitives, numOverlappingPrimitives, depth, edges,
		prims0, prims1, badRefines);
}

void CKdTreeAccelerator::InitializeInteriorNode(int node, const BBox& nodeBounds,
		const std::vector<BBox>& primitiveBounds, uint* overlappingPrimitives,
		int numOverlappingPrimitives, int depth, BoundEdge* edges[3],
		uint* prims0, uint* prims1, int badRefines)
{
	// Choose split axis and position for interior node
	int bestAxis = -1;
	int bestOffset = -1;
	float bestCost = INFINITY;
	float oldCost = m_IntersectionCost * float(numOverlappingPrimitives);
	float totalSA = nodeBounds.SurfaceArea();
	float invTotalSA = 1.f / totalSA;
	glm::vec3 d = nodeBounds.pMax - nodeBounds.pMin;

	uint axis = nodeBounds.MaximumExtent();
	int retries = 0;

	// label for jump to choose another split
	retrySplit:

	// Initialize edges for choosen axis
	for(int i = 0; i < numOverlappingPrimitives; ++i)
	{
		int primitive = overlappingPrimitives[i];
		const BBox& bbox = primitiveBounds[primitive];
		edges[axis][2 * i + 0] = BoundEdge(bbox.pMin[axis], primitive, true);
		edges[axis][2 * i + 1] = BoundEdge(bbox.pMax[axis], primitive, false);
	}
	std::sort(&edges[axis][0], &edges[axis][2*numOverlappingPrimitives]);

	// Compute cost of all splits for the choosen axis to find best
	int nBelow = 0;
	int nAbove = numOverlappingPrimitives;
	for(int i = 0; i < 2 * numOverlappingPrimitives; ++i)
	{
		if(edges[axis][i].type == BoundEdge::END) --nAbove;

		float split = edges[axis][i].t;
		if(split > nodeBounds.pMin[axis] && split < nodeBounds.pMax[axis])
		{
			// compute cost of split at i-th edge
			uint oA0 = (axis + 1) % 3; // first other axis 
			uint oA1 = (axis + 2) % 3; // second other axis
			float belowSA = 2 * (d[oA0] * d[oA1] + (split - nodeBounds.pMin[axis]) * (d[oA0] + d[oA1]));
			float aboveSA = 2 * (d[oA0] * d[oA1] + (nodeBounds.pMax[axis] - split) * (d[oA0] + d[oA1]));
			float pBelow = belowSA * invTotalSA;
			float pAbove = aboveSA * invTotalSA;
			float bonus = (nAbove==0 || nBelow==0) ? m_EmptyBonus : 0.f;
			float cost = m_IntersectionCost * (1.f - bonus) * (pBelow * nBelow + pAbove * nAbove)
				+ m_TraversalCost;

			// update best split if this split has lower costs
			if(cost < bestCost)
			{
				bestCost = cost;
				bestAxis = axis;
				bestOffset = i;
			}
		}

		if(edges[axis][i].type == BoundEdge::START) ++nBelow;
	}

	// try another axis of no good slit was found
	if(bestAxis == -1 && retries < 2)
	{
		++retries;
		axis = (axis + 1) % 3;
		goto retrySplit;
	}
	
	// Create lead if no good splits were found
	if (bestCost > oldCost) ++badRefines;
	if ((bestCost > 4.f * oldCost && numOverlappingPrimitives < 16) ||
		bestAxis == -1 || badRefines == 3) {
		m_Nodes[node].InitLeaf(overlappingPrimitives, numOverlappingPrimitives);
		return;
	}
	
	// Classify overlapping primitives with respect to split
	int n0 = 0, n1 = 0;
	for (int i = 0; i < bestOffset; ++i) {
		if (edges[bestAxis][i].type == BoundEdge::START) {
			prims0[n0++] = edges[bestAxis][i].primitive;
		}
	}
	for (int i = bestOffset+1; i < 2*numOverlappingPrimitives; ++i) {
		if (edges[bestAxis][i].type == BoundEdge::END) {
			prims1[n1++] = edges[bestAxis][i].primitive;
		}
	}

	// Recursively initialize child nodes
	float split = edges[bestAxis][bestOffset].t;
	BBox bounds0 = nodeBounds;
	BBox bounds1 = nodeBounds;
	bounds0.pMax[bestAxis] = split;
	bounds1.pMin[bestAxis] = split;

	BuildTreeRecursive(node + 1, bounds0, primitiveBounds, prims0, n0, depth - 1, 
		edges, prims0, prims1 + numOverlappingPrimitives, badRefines);

	uint aboveChild = m_NextFreeNode;
	m_Nodes[node].InitInterior(bestAxis, aboveChild, split);

	BuildTreeRecursive(aboveChild, bounds1, primitiveBounds, prims1, n1, depth - 1, 
		edges, prims0, prims1 + numOverlappingPrimitives, badRefines);
}

bool CKdTreeAccelerator::Intersect(const Ray& ray, float *t, Intersection* pIntersection) const
{
	// compute initial parametric range of ray inside kd-tree extent
	float t_min = std::numeric_limits<float>::max();
	float t_max = std::numeric_limits<float>::min();

	float t_best = std::numeric_limits<float>::max();
	Intersection isect_best;

	if(!m_BoundingBox.IntersectP(ray, &t_min, &t_max))
		return false;

	// prepare to traverse kd-tree for ray
	glm::vec3 invDir(1.f/ray.d.x, 1.f/ray.d.y, 1.f/ray.d.z);
	KdToTraverse toTraverse[MAX_TO_TRAVERSE];
	int toTraversePos = 0;

	// traverse kd-tree nodes in order for ray
	bool hit = false;
	const KdAccelNode* node = &m_Nodes[0];
	while(node != NULL)
	{
		// bail out if we found a hit closer than the current node
		if(t_best <= t_min) break;

		if(!node->IsLeaf())
		{
			// process kd-tree interior node

			// compute parametric distance along ray to split plane
			int axis = node->GetSplitAxis();
			
			// get node children pointers for ray
			const KdAccelNode* firstChild;
			const KdAccelNode* secondChild;
			
			int belowFirst = (ray.o[axis] < node->GetSplitPosition()) || 
				(ray.o[axis] == node->GetSplitPosition() && ray.d[axis] >= 0);
			
			if(belowFirst)
			{
				firstChild = node + 1;
				secondChild = &m_Nodes[node->AboveChild()];
			}
			else
			{
				firstChild = &m_Nodes[node->AboveChild()];
				secondChild = node + 1;
			}

			// advance to next child node, possibly enqueue other child

			// t for which ray.o + t * ray.d intersects the split plane
			float t_plane = (node->GetSplitPosition() - ray.o[axis]) * invDir[axis]; 

			if(t_plane > t_max || t_plane <= 0)
				node = firstChild;
			else if(t_plane < t_min)
				node = secondChild;
			else
			{
				// enqueue second child in todo list
				toTraverse[toTraversePos].node = secondChild;
				toTraverse[toTraversePos].t_min = t_plane;
				toTraverse[toTraversePos].t_max = t_max;
				toTraversePos++;

				node = firstChild;
				t_max = t_plane;
			}
		}
		else
		{
			// check for intersection inside leaf node
			uint numPrimitives = node->GetNumPrimitives();
			if(numPrimitives == 1)
			{
				CPrimitive* primitive = m_Primitives[node->m_OnePrimitive];
				float t_temp = 0.f;
				Intersection isect_temp;
				if(primitive->Intersect(ray, &t_temp, &isect_temp))
				{
					if(t_temp < t_best && t_temp > 0.f) {
						t_best = t_temp;
						isect_best = isect_temp;
						hit = true;
					}
				}
			}
			else
			{
				uint* primitives = node->m_Primitives;
				for(uint i = 0; i < node->GetNumPrimitives(); ++i)
				{
					CPrimitive* primitive = m_Primitives[node->m_Primitives[i]];
					float t_temp = 0.f;
					Intersection isect_temp;
					if(primitive->Intersect(ray, &t_temp, &isect_temp))
					{
						if(t_temp < t_best && t_temp > 0.f) {
							t_best = t_temp;
							isect_best = isect_temp;
							hit = true;
						}
					}
				}
			}

			// grab next node to process from toTraverse list
			if(toTraversePos > 0)
			{
				--toTraversePos;
				node = toTraverse[toTraversePos].node;
				t_min = toTraverse[toTraversePos].t_min;
				t_max = toTraverse[toTraversePos].t_max;
			}
			else
				break;
		}
	}

	*pIntersection = isect_best;
	*t = t_best; 

	return hit;
}

void CKdTreeAccelerator::PrintForDebug()
{
	for(int i = 0; i < m_NextFreeNode; ++i) {
		if(m_Nodes[i].IsLeaf())	{
			std::cout << "Node " << i << " is leaf" << std::endl;
			std::cout << "Overlapping primitives: " << std::endl;
			if(m_Nodes[i].GetNumPrimitives() == 1)
				std::cout << "Primitive Index: " << m_Nodes[i].m_OnePrimitive << std::endl;
			else 
				for(uint j = 0; j < m_Nodes[i].GetNumPrimitives(); ++j)
					std::cout << "Primitive Index: " << m_Nodes[i].m_Primitives[j] << std::endl;
		} else {
			std::cout << "Node " << i << " is inner node with childnodes " << (i+1) << " and " << m_Nodes[i].AboveChild() << std::endl;
			std::cout << "Split Axis: " << m_Nodes[i].GetSplitAxis() << ", split position: " << m_Nodes[i].GetSplitPosition() << std::endl;
		}
	}
}

std::vector<CPrimitive*> CKdTreeAccelerator::GetPrimitivesOfNode(int i)
{
	std::vector<CPrimitive*> primitives;
	if(i < m_NextFreeNode)
	{
		if(m_Nodes[i].IsLeaf())
		{
			if(m_Nodes[i].GetNumPrimitives() == 1)
			{
				primitives.push_back(m_Primitives[m_Nodes[i].m_OnePrimitive]);
			}
			else
			{
				for(uint j = 0; j < m_Nodes[i].GetNumPrimitives(); ++j)
				{
					primitives.push_back(m_Primitives[m_Nodes[i].m_Primitives[j]]);
				}
			}
		}
	}
	return primitives;
}

void KdAccelNode::InitLeaf(uint* primitives, int numPrimitives)
{
	m_Flags = 3;
	m_NumPrimitives |= (numPrimitives << 2);

	if(numPrimitives == 0)
		m_OnePrimitive = 0;
	else if(numPrimitives == 1)
		m_OnePrimitive = primitives[0];
	else {
		m_Primitives = new uint[numPrimitives];
		for(int i = 0; i < numPrimitives; ++i)
			m_Primitives[i] = primitives[i];
	}
}

void KdAccelNode::InitInterior(uint splitAxis, uint aboveChild, float splitPosition)
{
	m_SplitPosition = splitPosition;
	m_Flags = splitAxis;
	m_AboveChild |= (aboveChild << 2);
}

KdAccelNode::~KdAccelNode()
{
	if(GetNumPrimitives() > 1)
		delete [] m_Primitives;
}