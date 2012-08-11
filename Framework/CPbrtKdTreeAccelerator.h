#ifndef _C_PBRT_KD_TREE_ACCELERATOR_H_
#define _C_PBRT_KD_TREE_ACCELERATOR_H_

#include "CPrimitive.h"

#include "BBox.h"
#include "Ray.h"

#include <vector>

typedef unsigned int uint;

// KdTreeAccel Declarations
struct KdAccelNode;
struct BoundEdge;

class CPbrtKdTreeAccel
{
public:
    CPbrtKdTreeAccel(const std::vector<CPrimitive*>& p, int icost = 80, 
		int scost = 1, float ebonus = 0.5f, int maxp = 1, int maxDepth = -1);
    ~CPbrtKdTreeAccel();
	
	BBox WorldBound() const { return bounds; }
    
	bool Intersect(const Ray &ray, Intersection *isect, CPrimitive::IsectMode isectMode) const;

private:
    // KdTreeAccel Private Methods
    void buildTree(int nodeNum, const BBox &bounds,
        const std::vector<BBox> &primBounds, uint *primNums, int nprims, int depth,
        BoundEdge *edges[3], uint *prims0, uint *prims1, int badRefines = 0);

    // KdTreeAccel Private Data
    int isectCost, traversalCost, maxPrims, maxDepth;
    float emptyBonus;
    std::vector<CPrimitive*> primitives;
    KdAccelNode *nodes;
    int nAllocedNodes, nextFreeNode;
    BBox bounds;
};

struct KdToDo {
    const KdAccelNode *node;
    float tmin, tmax;
};

#endif // _C_PBRT_KD_TREE_ACCELERATOR_H_