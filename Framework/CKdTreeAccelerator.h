#ifndef _C_KD_TREE_ACCELERATOR_H_
#define _C_KD_TREE_ACCELERATOR_H_

class CPrimitive;

#include "BBox.h"
#include "Ray.h"
#include "Intersection.h"
#include "CPrimitive.h"

#include <vector>

struct KdAccelNode;
struct BoundEdge;
class Intersection;

typedef unsigned int uint;

class CKdTreeAccelerator
{
public:
	CKdTreeAccelerator(const std::vector<CPrimitive*>& primitives,
		int intersectionCost, int traversalCost, int maxNumPrimitives, int maxDepth);
	~CKdTreeAccelerator();

	void BuildTree();

	bool Intersect(const Ray& ray, float* t, Intersection* pIntersection, CPrimitive::IsectMode isectMode) const;

	void PrintForDebug();

	int GetNumNodes() { return m_NextFreeNode; }
	KdAccelNode* GetNodes() { return m_Nodes; }
	std::vector<CPrimitive*> GetPrimitivesOfNode(int i);

private:
	void BuildTreeRecursive(int node, const BBox& nodeBounds,
		const std::vector<BBox>& primitiveBounds, uint* overlappingPrimitives,
		int numOverlappingPrimitives, int depth, BoundEdge* edges[3],
		uint* prims0, uint* prims1, int badRefines);

	void InitializeInteriorNode(int node, const BBox& nodeBounds,
		const std::vector<BBox>& primitiveBounds, uint* overlappingPrimitives,
		int numOverlappingPrimitives, int depth, BoundEdge* edges[3],
		uint* prims0, uint* prims1, int badRefines);

	int m_IntersectionCost;
	int m_TraversalCost;
	int m_MaxNumPrimitives;		// maximum number of primitives per leaf
	int m_MaxDepth;

	float m_EmptyBonus;

	KdAccelNode* m_Nodes;		// array where all the nodes are stored
	int m_NumAllocatedNodes;	// the size of the nodes array
	int m_NextFreeNode;			// index of the next free slot in the nodes array

	BBox m_BoundingBox;

	std::vector<CPrimitive*> m_Primitives;
};

struct KdAccelNode
{
public:
	KdAccelNode() { m_NumPrimitives = 0; m_Primitives = 0; }
	~KdAccelNode();

	void InitLeaf(uint* primitives, int numPrimitives);
	void InitInterior(uint splitAxis, uint aboveChild, float splitPosition);

	float GetSplitPosition() const { return m_SplitPosition; }
	uint GetNumPrimitives() const { return m_NumPrimitives >> 2; }
	uint GetSplitAxis() const { return m_Flags & 3; }
	bool IsLeaf() const { return (m_Flags & 3) == 3; }
	uint AboveChild() const { return m_AboveChild >> 2; }

	union {
		float m_SplitPosition;	// the position of the split on the split axis
		uint m_OnePrimitive;	// if there is just one primitive intersecting this node its index relative to m_Primitives is stored
		uint* m_Primitives;		// if more than one primitives intersect this node their indices are stored in this dynamically allocated memory
	};

	union {
		uint m_Flags;			// stores split axis in the 2 lowest bits and the number of primitives in the upper 30bits.
		uint m_NumPrimitives;
		uint m_AboveChild;		// one child is always right next to the node in the node array. This is the index of the other child in the node array.
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
	BoundEdge(float t_param, int primitive_param, bool starting) 
	{
		t = t_param;
		primitive = primitive_param;
		type = starting ? START : END;
	}
	
	bool operator<(const BoundEdge& e) const 
	{
		if(t == e.t)
			return (int)type < (int)e.type;
		else
			return t < e.t;
	}

	float t;
	int primitive;
	enum {START, END} type;
};

struct KdToTraverse
{
	const KdAccelNode* node;
	float t_min, t_max;
};

#endif _C_KD_TREE_ACCELERATOR_H_