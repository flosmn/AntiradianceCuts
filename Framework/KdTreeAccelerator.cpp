#include "KdTreeAccelerator.h"

#include "Triangle.h"

#include <algorithm>
#include <limits>
#include <iostream>

static const float INFINITY = std::numeric_limits<float>::max();
static const int MAX_TO_TRAVERSE = 64;

struct KdAccelNode;

KdTreeAccelerator::KdTreeAccelerator(const std::vector<Triangle>& primitives,
	int intersectionCost, int traversalCost, int maxNumPrimitives, int maxDepth)
	: m_intersectionCost(intersectionCost), m_traversalCost(traversalCost),
	m_maxNumPrimitives(maxNumPrimitives), m_maxDepth(maxDepth)

{
	m_primitives = primitives;

	m_nextFreeNode = 0;
	m_numAllocatedNodes = 0;
	m_emptyBonus = 0.f;
	m_nodes = 0;

	if(m_maxDepth <= 0)
		m_maxDepth = int(8.f + 1.3f * glm::log2(float(m_primitives.size())));
}

KdTreeAccelerator::~KdTreeAccelerator()
{
}

void KdTreeAccelerator::buildTree()
{
	// compute bounds for kd-tree construction
	std::vector<BBox> primitiveBounds;
	primitiveBounds.reserve(m_primitives.size());
	for(uint i = 0; i < m_primitives.size(); ++i)
	{
		Triangle& prim = m_primitives[i];
		BBox b = prim.getBBox();
		m_boundingBox = BBox::Union(m_boundingBox, b);
		primitiveBounds.push_back(b);
	}

	// allocate memory for construction
	BoundEdge* edges[3];
	for(int i = 0; i < 3; ++i)
		edges[i] = new BoundEdge[2 * m_primitives.size()];
	uint* prims0 = new uint[m_primitives.size()];
	uint* prims1 = new uint[(m_maxDepth+1) * m_primitives.size()];

	// initialize overlapping primitives (all at the beginning)
	uint* overlappingPrimitives = new uint[m_primitives.size()];
	for(uint i = 0; i < m_primitives.size(); ++i)
	{
		overlappingPrimitives[i] = i;
	}

	// start to recursively build the tree
	buildTreeRecursive(0, m_boundingBox, primitiveBounds, overlappingPrimitives,
		(int)m_primitives.size(), m_maxDepth, edges, prims0, prims1, 0);

	// release allocated memory
	for(int i = 0; i < 3; ++i)
		delete [] edges[i];
	delete [] prims0;
	delete [] prims1;
}

void KdTreeAccelerator::buildTreeRecursive(int node, const BBox& nodeBounds,
		const std::vector<BBox>& primitiveBounds, uint* overlappingPrimitives,
		int numOverlappingPrimitives, int depth, BoundEdge* edges[3],
		uint* prims0, uint* prims1, int badRefines)
{
	// get next free node from nodes array
	if(m_nextFreeNode == m_numAllocatedNodes)
	{
		int newNumAllocatedNodes = std::max(2 * m_numAllocatedNodes, 4096);
		KdAccelNode* n = new KdAccelNode[newNumAllocatedNodes];
		memset(n, 0, newNumAllocatedNodes * sizeof(KdAccelNode));

		if(m_numAllocatedNodes > 0)
		{
			memcpy(n, m_nodes, m_numAllocatedNodes * sizeof(KdAccelNode));
			//delete [] m_nodes;			
		}
		
		m_nodes = n;
		m_numAllocatedNodes = newNumAllocatedNodes;
	}
	++m_nextFreeNode;

	// Initialize leaf node of termination criteria is met
	if(numOverlappingPrimitives <= m_maxNumPrimitives || depth == 0)
	{
		m_nodes[node].initLeaf(overlappingPrimitives, numOverlappingPrimitives);
	}

	// Initialize interior node and continue recursion
	initializeInteriorNode(node, nodeBounds, primitiveBounds, 
		overlappingPrimitives, numOverlappingPrimitives, depth, edges,
		prims0, prims1, badRefines);
}

void KdTreeAccelerator::initializeInteriorNode(int node, const BBox& nodeBounds,
		const std::vector<BBox>& primitiveBounds, uint* overlappingPrimitives,
		int numOverlappingPrimitives, int depth, BoundEdge* edges[3],
		uint* prims0, uint* prims1, int badRefines)
{
	// Choose split axis and position for interior node
	int bestAxis = -1;
	int bestOffset = -1;
	float bestCost = INFINITY;
	float oldCost = m_intersectionCost * float(numOverlappingPrimitives);
	float totalSA = nodeBounds.getSurfaceArea();
	float invTotalSA = 1.f / totalSA;
	glm::vec3 d = nodeBounds.getMax() - nodeBounds.getMin();

	uint axis = nodeBounds.getAxisMaximumExtent();
	int retries = 0;

	// label for jump to choose another split
	retrySplit:

	// Initialize edges for choosen axis
	for(int i = 0; i < numOverlappingPrimitives; ++i)
	{
		int primitive = overlappingPrimitives[i];
		const BBox& bbox = primitiveBounds[primitive];
		edges[axis][2 * i + 0] = BoundEdge(bbox.getMin()[axis], primitive, true);
		edges[axis][2 * i + 1] = BoundEdge(bbox.getMax()[axis], primitive, false);
	}
	std::sort(&edges[axis][0], &edges[axis][2*numOverlappingPrimitives]);

	// Compute cost of all splits for the choosen axis to find best
	int nBelow = 0;
	int nAbove = numOverlappingPrimitives;
	for(int i = 0; i < 2 * numOverlappingPrimitives; ++i)
	{
		if(edges[axis][i].m_type == BoundEdge::END) --nAbove;

		float split = edges[axis][i].m_t;
		if(split > nodeBounds.getMin()[axis] && split < nodeBounds.getMax()[axis])
		{
			// compute cost of split at i-th edge
			uint oA0 = (axis + 1) % 3; // first other axis 
			uint oA1 = (axis + 2) % 3; // second other axis
			float belowSA = 2 * (d[oA0] * d[oA1] + (split - nodeBounds.getMin()[axis]) * (d[oA0] + d[oA1]));
			float aboveSA = 2 * (d[oA0] * d[oA1] + (nodeBounds.getMax()[axis] - split) * (d[oA0] + d[oA1]));
			float pBelow = belowSA * invTotalSA;
			float pAbove = aboveSA * invTotalSA;
			float bonus = (nAbove==0 || nBelow==0) ? m_emptyBonus : 0.f;
			float cost = m_intersectionCost * (1.f - bonus) * (pBelow * nBelow + pAbove * nAbove)
				+ m_traversalCost;

			// update best split if this split has lower costs
			if(cost < bestCost)
			{
				bestCost = cost;
				bestAxis = axis;
				bestOffset = i;
			}
		}

		if(edges[axis][i].m_type == BoundEdge::START) ++nBelow;
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
		m_nodes[node].initLeaf(overlappingPrimitives, numOverlappingPrimitives);
		return;
	}
	
	// Classify overlapping primitives with respect to split
	int n0 = 0, n1 = 0;
	for (int i = 0; i < bestOffset; ++i) {
		if (edges[bestAxis][i].m_type == BoundEdge::START) {
			prims0[n0++] = edges[bestAxis][i].m_primitive;
		}
	}
	for (int i = bestOffset+1; i < 2*numOverlappingPrimitives; ++i) {
		if (edges[bestAxis][i].m_type == BoundEdge::END) {
			prims1[n1++] = edges[bestAxis][i].m_primitive;
		}
	}

	// Recursively initialize child nodes
	float split = edges[bestAxis][bestOffset].m_t;
	glm::vec3 min = nodeBounds.getMax();
	glm::vec3 max = nodeBounds.getMin();
	min[bestAxis] = split;
	max[bestAxis] = split;
	BBox bounds0 = nodeBounds;
	BBox bounds1 = nodeBounds;
	bounds0.setMax(max);
	bounds1.setMin(min);

	buildTreeRecursive(node + 1, bounds0, primitiveBounds, prims0, n0, depth - 1, 
		edges, prims0, prims1 + numOverlappingPrimitives, badRefines);

	uint aboveChild = m_nextFreeNode;
	m_nodes[node].initInterior(bestAxis, aboveChild, split);

	buildTreeRecursive(aboveChild, bounds1, primitiveBounds, prims1, n1, depth - 1, 
		edges, prims0, prims1 + numOverlappingPrimitives, badRefines);
}

bool KdTreeAccelerator::intersect(const Ray& ray, float *t, Intersection* pIntersection, Triangle::IsectMode isectMode) const
{
	// compute initial parametric range of ray inside kd-tree extent
	float t_min = std::numeric_limits<float>::max();
	float t_max = std::numeric_limits<float>::min();

	float t_best = std::numeric_limits<float>::max();
	Intersection isect_best;

	if(!m_boundingBox.intersect(ray, &t_min, &t_max))
		return false;

	// prepare to traverse kd-tree for ray
	glm::vec3 invDir(1.f/ray.getDirection().x, 1.f/ray.getDirection().y, 1.f/ray.getDirection().z);
	KdToTraverse toTraverse[MAX_TO_TRAVERSE];
	int toTraversePos = 0;

	// traverse kd-tree nodes in order for ray
	bool hit = false;
	const KdAccelNode* node = &m_nodes[0];
	while(node != NULL)
	{
		// bail out if we found a hit closer than the current node
		if(t_best <= t_min) break;

		if(!node->isLeaf())
		{
			// process kd-tree interior node

			// compute parametric distance along ray to split plane
			const int axis = node->getSplitAxis();
			
			// get node children pointers for ray
			const KdAccelNode* firstChild;
			const KdAccelNode* secondChild;
			
			const int belowFirst = (ray.getOrigin()[axis] < node->getSplitPosition()) || 
				(ray.getOrigin()[axis] == node->getSplitPosition() && ray.getDirection()[axis] >= 0);
			
			if(belowFirst)
			{
				firstChild = node + 1;
				secondChild = &m_nodes[node->aboveChild()];
			}
			else
			{
				firstChild = &m_nodes[node->aboveChild()];
				secondChild = node + 1;
			}

			// advance to next child node, possibly enqueue other child

			// t for which ray.getOrigin() + t * ray.getDirection() intersects the split plane
			float t_plane = (node->getSplitPosition() - ray.getOrigin()[axis]) * invDir[axis]; 

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
			const uint numPrimitives = node->getNumPrimitives();
			if(numPrimitives == 1)
			{
				Triangle const& primitive = m_primitives[node->m_onePrimitive];
				float t_temp = 0.f;
				Intersection isect_temp;
				if(primitive.intersect(ray, &t_temp, &isect_temp, isectMode))
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
				uint* primitives = node->m_primitives;
				for(uint i = 0; i < node->getNumPrimitives(); ++i)
				{
					Triangle const& primitive = m_primitives[node->m_primitives[i]];
					float t_temp = 0.f;
					Intersection isect_temp;
					if(primitive.intersect(ray, &t_temp, &isect_temp, isectMode))
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

void KdTreeAccelerator::printForDebug()
{
	for(int i = 0; i < m_nextFreeNode; ++i) {
		if(m_nodes[i].isLeaf())	{
			std::cout << "Node " << i << " is leaf" << std::endl;
			std::cout << "Overlapping primitives: " << std::endl;
			if(m_nodes[i].getNumPrimitives() == 1)
				std::cout << "Primitive Index: " << m_nodes[i].m_onePrimitive << std::endl;
			else 
				for(uint j = 0; j < m_nodes[i].getNumPrimitives(); ++j)
					std::cout << "Primitive Index: " << m_nodes[i].m_primitives[j] << std::endl;
		} else {
			std::cout << "Node " << i << " is inner node with childnodes " << (i+1) << " and " << m_nodes[i].aboveChild() << std::endl;
			std::cout << "Split Axis: " << m_nodes[i].getSplitAxis() << ", split position: " << m_nodes[i].getSplitPosition() << std::endl;
		}
	}
}

std::vector<Triangle> KdTreeAccelerator::getPrimitivesOfNode(int i)
{
	std::vector<Triangle> primitives;
	if(i < m_nextFreeNode)
	{
		if(m_nodes[i].isLeaf())
		{
			if(m_nodes[i].getNumPrimitives() == 1)
			{
				primitives.push_back(m_primitives[m_nodes[i].m_onePrimitive]);
			}
			else
			{
				for(uint j = 0; j < m_nodes[i].getNumPrimitives(); ++j)
				{
					primitives.push_back(m_primitives[m_nodes[i].m_primitives[j]]);
				}
			}
		}
	}
	return primitives;
}

void KdAccelNode::initLeaf(uint* primitives, int numPrimitives)
{
	m_flags = 3;
	m_numPrimitives |= (numPrimitives << 2);

	if(numPrimitives == 0)
		m_onePrimitive = 0;
	else if(numPrimitives == 1)
		m_onePrimitive = primitives[0];
	else {
		m_primitives = new uint[numPrimitives];
		for(int i = 0; i < numPrimitives; ++i)
			m_primitives[i] = primitives[i];
	}
}

void KdAccelNode::initInterior(uint splitAxis, uint aboveChild, float splitPosition)
{
	m_splitPosition = splitPosition;
	m_flags = splitAxis;
	m_aboveChild |= (aboveChild << 2);
}

KdAccelNode::~KdAccelNode()
{
	if(getNumPrimitives() > 1 && m_primitives)
		delete [] m_primitives;
}
