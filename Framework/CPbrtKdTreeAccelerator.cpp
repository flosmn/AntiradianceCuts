#include "CPbrtKdTreeAccelerator.h"

#include "Intersection.h"

#include <assert.h>
#include <algorithm>
#include <limits>

#include <glm/glm.hpp>

const float INFINITY = std::numeric_limits<float>::max();
const uint MAX_TODO = 64;

inline int Floor2Int(float val) {
    return (int)floorf(val);
}

inline int Round2Int(float val) {
    return Floor2Int(val + 0.5f);
}

inline float Log2(float x) {
    static float invLog2 = 1.f / logf(2.f);
    return logf(x) * invLog2;
}

inline int Log2Int(float v) {
    return Floor2Int(Log2(v));
}

struct KdAccelNode {
    void initLeaf(uint *primNums, int np);
    void initInterior(uint axis, uint ac, float s) {
        split = s;
        flags = axis;
        aboveChild |= (ac << 2);
    }
    float SplitPos() const { return split; }
    uint nPrimitives() const { return nPrims >> 2; }
    uint SplitAxis() const { return flags & 3; }
    bool IsLeaf() const { return (flags & 3) == 3; }
    uint AboveChild() const { return aboveChild >> 2; }
    union {
        float split;        // Interior
        uint onePrimitive;  // Leaf
        uint *primitives;   // Leaf
    };

private:
    union {
        uint flags;         // Both
        uint nPrims;        // Leaf
        uint aboveChild;    // Interior
    };
};

void KdAccelNode::initLeaf(uint *primNums, int np) {
    flags = 3;
    nPrims |= (np << 2);
    // Store primitive ids for leaf node
    if (np == 0)
        onePrimitive = 0;
    else if (np == 1)
        onePrimitive = primNums[0];
    else {
        primitives = new uint[np];
        for (int i = 0; i < np; ++i)
            primitives[i] = primNums[i];
    }
}


struct BoundEdge {
    // BoundEdge Public Methods
    BoundEdge() { }
    BoundEdge(float tt, int pn, bool starting) {
        t = tt;
        primNum = pn;
        type = starting ? START : END;
    }
    bool operator<(const BoundEdge &e) const {
        if (t == e.t)
            return (int)type < (int)e.type;
        else return t < e.t;
    }
    float t;
    int primNum;
    enum { START, END } type;
};



// KdTreeAccel Method Definitions
CPbrtKdTreeAccel::CPbrtKdTreeAccel(const std::vector<CPrimitive*>& p,
                         int icost, int tcost, float ebonus, int maxp,
                         int md)
    : isectCost(icost), traversalCost(tcost), maxPrims(maxp), maxDepth(md),
      emptyBonus(ebonus) 
{
	primitives = p;
    
	nextFreeNode = nAllocedNodes = 0;
    if (maxDepth <= 0)
        maxDepth = Round2Int(8 + 1.3f * Log2Int(float(primitives.size())));

    // Compute bounds for kd-tree construction
    std::vector<BBox> primBounds;
    primBounds.reserve(primitives.size());
    for (uint i = 0; i < primitives.size(); ++i) {
        BBox b = primitives[i]->GetBBox();
        bounds = BBox::Union(bounds, b);
        primBounds.push_back(b);
    }

    // Allocate working memory for kd-tree construction
    BoundEdge *edges[3];
    for (int i = 0; i < 3; ++i)
        edges[i] = new BoundEdge[2*primitives.size()];
    uint *prims0 = new uint[primitives.size()];
    uint *prims1 = new uint[(maxDepth+1) * primitives.size()];

    // Initialize _primNums_ for kd-tree construction
    uint *primNums = new uint[primitives.size()];
    for (uint i = 0; i < primitives.size(); ++i)
        primNums[i] = i;

    // Start recursive construction of kd-tree
    buildTree(0, bounds, primBounds, primNums, (uint)primitives.size(),
              maxDepth, edges, prims0, prims1);

    // Free working memory for kd-tree construction
    delete[] primNums;
    for (int i = 0; i < 3; ++i)
        delete[] edges[i];
    delete[] prims0;
    delete[] prims1;
}

CPbrtKdTreeAccel::~CPbrtKdTreeAccel() {
    delete [] nodes;
}

void CPbrtKdTreeAccel::buildTree(int nodeNum, const BBox &nodeBounds,
        const std::vector<BBox> &allPrimBounds, uint *primNums,
        int nPrimitives, int depth, BoundEdge *edges[3],
        uint *prims0, uint *prims1, int badRefines) 
{
	assert(nodeNum == nextFreeNode);
    // Get next free node from _nodes_ array
    if (nextFreeNode == nAllocedNodes) {
        int nAlloc = std::max(2 * nAllocedNodes, 512);
        KdAccelNode *n = new KdAccelNode[nAlloc];
        if (nAllocedNodes > 0) {
            memcpy(n, nodes, nAllocedNodes * sizeof(KdAccelNode));
            delete [] nodes;
        }
        nodes = n;
        nAllocedNodes = nAlloc;
    }
    ++nextFreeNode;

    // Initialize leaf node if termination criteria met
    if (nPrimitives <= maxPrims || depth == 0) {
        nodes[nodeNum].initLeaf(primNums, nPrimitives);
        return;
    }

    // Initialize interior node and continue recursion

    // Choose split axis position for interior node
    int bestAxis = -1, bestOffset = -1;
    float bestCost = INFINITY;
    float oldCost = isectCost * float(nPrimitives);
    float totalSA = nodeBounds.SurfaceArea();
    float invTotalSA = 1.f / totalSA;
    glm::vec3 d = nodeBounds.pMax - nodeBounds.pMin;

    // Choose which axis to split along
    uint axis = nodeBounds.MaximumExtent();
    int retries = 0;
    retrySplit:

    // Initialize edges for _axis_
    for (int i = 0; i < nPrimitives; ++i) {
        int pn = primNums[i];
        const BBox &bbox = allPrimBounds[pn];
        edges[axis][2*i] =   BoundEdge(bbox.pMin[axis], pn, true);
        edges[axis][2*i+1] = BoundEdge(bbox.pMax[axis], pn, false);
    }
   std::sort(&edges[axis][0], &edges[axis][2*nPrimitives]);

    // Compute cost of all splits for _axis_ to find best
    int nBelow = 0, nAbove = nPrimitives;
    for (int i = 0; i < 2*nPrimitives; ++i) {
        if (edges[axis][i].type == BoundEdge::END) --nAbove;
        float edget = edges[axis][i].t;
        if (edget > nodeBounds.pMin[axis] &&
            edget < nodeBounds.pMax[axis]) {
            // Compute cost for split at _i_th edge
            uint otherAxis0 = (axis + 1) % 3, otherAxis1 = (axis + 2) % 3;
            float belowSA = 2 * (d[otherAxis0] * d[otherAxis1] +
                                 (edget - nodeBounds.pMin[axis]) *
                                 (d[otherAxis0] + d[otherAxis1]));
            float aboveSA = 2 * (d[otherAxis0] * d[otherAxis1] +
                                 (nodeBounds.pMax[axis] - edget) *
                                 (d[otherAxis0] + d[otherAxis1]));
            float pBelow = belowSA * invTotalSA;
            float pAbove = aboveSA * invTotalSA;
            float eb = (nAbove == 0 || nBelow == 0) ? emptyBonus : 0.f;
            float cost = traversalCost +
                         isectCost * (1.f - eb) * (pBelow * nBelow + pAbove * nAbove);

            // Update best split if this is lowest cost so far
            if (cost < bestCost)  {
                bestCost = cost;
                bestAxis = axis;
                bestOffset = i;
            }
        }
        if (edges[axis][i].type == BoundEdge::START) ++nBelow;
    }
    assert(nBelow == nPrimitives && nAbove == 0);

    // Create leaf if no good splits were found
    if (bestAxis == -1 && retries < 2) {
        ++retries;
        axis = (axis+1) % 3;
        goto retrySplit;
    }
    if (bestCost > oldCost) ++badRefines;
    if ((bestCost > 4.f * oldCost && nPrimitives < 16) ||
        bestAxis == -1 || badRefines == 3) {
        nodes[nodeNum].initLeaf(primNums, nPrimitives);
        return;
    }

    // Classify primitives with respect to split
    int n0 = 0, n1 = 0;
    for (int i = 0; i < bestOffset; ++i)
        if (edges[bestAxis][i].type == BoundEdge::START)
            prims0[n0++] = edges[bestAxis][i].primNum;

    for (int i = bestOffset+1; i < 2*nPrimitives; ++i)
        if (edges[bestAxis][i].type == BoundEdge::END)
            prims1[n1++] = edges[bestAxis][i].primNum;

    // Recursively initialize children nodes
    float tsplit = edges[bestAxis][bestOffset].t;
    BBox bounds0 = nodeBounds, bounds1 = nodeBounds;
    bounds0.pMax[bestAxis] = bounds1.pMin[bestAxis] = tsplit;
    
	buildTree(nodeNum+1, bounds0,
              allPrimBounds, prims0, n0, depth-1, edges,
              prims0, prims1 + nPrimitives, badRefines);
    
	uint aboveChild = nextFreeNode;
    nodes[nodeNum].initInterior(bestAxis, aboveChild, tsplit);
    
	buildTree(aboveChild, bounds1, allPrimBounds, prims1, n1,
              depth-1, edges, prims0, prims1 + nPrimitives, badRefines);
}


bool CPbrtKdTreeAccel::Intersect(const Ray &ray,
	Intersection *isect, CPrimitive::IsectMode isectMode) const {
    // Compute initial parametric range of ray inside kd-tree extent
	float t_best = std::numeric_limits<float>::max();
	Intersection isect_best;

    float tmin, tmax;
    if (!bounds.IntersectP(ray, &tmin, &tmax))
    {
		return false;
    }

    // Prepare to traverse kd-tree for ray
    glm::vec3 invDir(1.f/ray.d.x, 1.f/ray.d.y, 1.f/ray.d.z);

    KdToDo todo[MAX_TODO];
    int todoPos = 0;

    // Traverse kd-tree nodes in order for ray
    bool hit = false;
    const KdAccelNode *node = &nodes[0];
    while (node != NULL) {
        // Bail out if we found a hit closer than the current node
		//if (t_best < tmin) break;
        if (!node->IsLeaf()) {
            // Process kd-tree interior node

            // Compute parametric distance along ray to split plane
            int axis = node->SplitAxis();
            float tplane = (node->SplitPos() - ray.o[axis]) * invDir[axis];

            // Get node children pointers for ray
            const KdAccelNode *firstChild, *secondChild;
            int belowFirst = (ray.o[axis] <  node->SplitPos()) ||
                             (ray.o[axis] == node->SplitPos() && ray.d[axis] <= 0);
            if (belowFirst) {
                firstChild = node + 1;
                secondChild = &nodes[node->AboveChild()];
            }
            else {
                firstChild = &nodes[node->AboveChild()];
                secondChild = node + 1;
            }

            // Advance to next child node, possibly enqueue other child
            if (tplane > tmax || tplane <= 0)
                node = firstChild;
            else if (tplane < tmin)
                node = secondChild;
            else {
                // Enqueue _secondChild_ in todo list
                todo[todoPos].node = secondChild;
                todo[todoPos].tmin = tplane;
                todo[todoPos].tmax = tmax;
                ++todoPos;
                node = firstChild;
                tmax = tplane;
            }
        }
        else {
            // Check for intersections inside leaf node
            uint nPrimitives = node->nPrimitives();
            if (nPrimitives == 1) {
                CPrimitive* prim = primitives[node->onePrimitive];
                // Check one primitive inside leaf node
				float t = 0.f;
                if (prim->Intersect(ray, &t, isect, isectMode))
                {
                    if (t < t_best && t > 0.f)
					{
						hit = true;
						t_best = t;
						isect_best = *isect;
					}					
                }
            }
            else {
                uint *prims = node->primitives;
                for (uint i = 0; i < nPrimitives; ++i) {
                    CPrimitive* prim = primitives[prims[i]];
                    // Check one primitive inside leaf node
                    float t = 0.f;
					if (prim->Intersect(ray, &t, isect, isectMode))
                    {
						if(t < t_best && t > 0.f)
						{
							hit = true;
							t_best = t;
							isect_best = *isect;
						}
                    }
                }
            }

            // Grab next node to process from todo list
            if (todoPos > 0) {
                --todoPos;
                node = todo[todoPos].node;
                tmin = todo[todoPos].tmin;
                tmax = todo[todoPos].tmax;
            }
            else
                break;
        }
    }

	*isect = isect_best;

    return hit;
}