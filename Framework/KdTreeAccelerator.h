#ifndef _KDTREEACCELERATOR_H_
#define _KDTREECCELERATOR_H_

#include "BBox.h"
#include "Ray.h"
#include "Intersection.h"
#include "Triangle.h"

#include <vector>

struct KdAccelNode;
struct BoundEdge;
class Intersection;

typedef unsigned int uint;

class KdTreeAccelerator
{
public:
	KdTreeAccelerator(const std::vector<Triangle>& primitives,
		int intersectionCost, int traversalCost, int maxNumPrimitives, int maxDepth);
	~KdTreeAccelerator();

	void buildTree();

	bool intersect(const Ray& ray, float* t, Intersection* pIntersection, Triangle::IsectMode isectMode) const;

	void printForDebug();

	int getNumNodes() { return m_nextFreeNode; }
	KdAccelNode* getNodes() { return m_nodes; }
	std::vector<Triangle> getPrimitivesOfNode(int i);

private:
	void buildTreeRecursive(int node, const BBox& nodeBounds,
		const std::vector<BBox>& primitiveBounds, uint* overlappingPrimitives,
		int numOverlappingPrimitives, int depth, BoundEdge* edges[3],
		uint* prims0, uint* prims1, int badRefines);

	void initializeInteriorNode(int node, const BBox& nodeBounds,
		const std::vector<BBox>& primitiveBounds, uint* overlappingPrimitives,
		int numOverlappingPrimitives, int depth, BoundEdge* edges[3],
		uint* prims0, uint* prims1, int badRefines);

	int m_intersectionCost;
	int m_traversalCost;
	int m_maxNumPrimitives;		// maximum number of primitives per leaf
	int m_maxDepth;

	float m_emptyBonus;

	KdAccelNode* m_nodes;		// array where all the nodes are stored
	int m_numAllocatedNodes;	// the size of the nodes array
	int m_nextFreeNode;			// index of the next free slot in the nodes array

	BBox m_boundingBox;

	std::vector<Triangle> m_primitives;
};

struct KdAccelNode
{
public:
	KdAccelNode() { m_numPrimitives = 0; m_primitives = 0; }
	~KdAccelNode();

	void initLeaf(uint* primitives, int numPrimitives);
	void initInterior(uint splitAxis, uint aboveChild, float splitPosition);

	float getSplitPosition() const { return m_splitPosition; }
	uint getNumPrimitives() const { return m_numPrimitives >> 2; }
	uint getSplitAxis() const { return m_flags & 3; }
	bool isLeaf() const { return (m_flags & 3) == 3; }
	uint aboveChild() const { return m_aboveChild >> 2; }

	union {
		float m_splitPosition;	// the position of the split on the split axis
		uint m_onePrimitive;	// if there is just one primitive intersecting this node its index relative to m_Primitives is stored
		uint* m_primitives;		// if more than one primitives intersect this node their indices are stored in this dynamically allocated memory
	};

	union {
		uint m_flags;			// stores split axis in the 2 lowest bits and the number of primitives in the upper 30bits.
		uint m_numPrimitives;
		uint m_aboveChild;		// one child is always right next to the node in the node array. This is the index of the other child in the node array.
	};
};


/* 
	Represents an edge (face) of the bounding box of a primitive.
	It is used when searching for a suitable split of an interior node.
*/
struct BoundEdge
{
public:
	BoundEdge() { }
	BoundEdge(float t, int primitive, bool starting) 
		: m_t(t), m_primitive(primitive) 
	{
		m_type = starting ? START : END;
	}
	
	bool operator<(const BoundEdge& e) const 
	{
		if(m_t == e.m_t)
			return (int)m_type < (int)e.m_type;
		else
			return m_t < e.m_t;
	}

	float m_t;
	int m_primitive;
	enum {START, END} m_type;
};

struct KdToTraverse
{
	const KdAccelNode* node;
	float t_min, t_max;
};

#endif // KDTREEACCELERATOR_H_
