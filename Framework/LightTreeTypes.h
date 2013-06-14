#ifndef _LIGHT_TREE_TYPES_H_
#define _LIGHT_TREE_TYPES_H_

#include <glm/glm.hpp>

#include "BBox.h"
#include "Structs.h"

#include <unordered_set>

typedef unsigned int uint;

struct CLUSTER_PAIR;
struct CLUSTER;

struct CLUSTER
{
	CLUSTER() {
		left = 0;
		right = 0;
	}

	uint id;
	int avplIndex;

	CLUSTER* left;
	CLUSTER* right;

	uint size;
	uint depth;
	BBox bbox;
	glm::vec3 mean;
	glm::vec3 intensity;
	glm::vec3 normal;

	glm::vec3 incomingDirection;
	float materialIndex;
	
	void Fill(CLUSTER_BUFFER* buffer)
	{
		buffer->avplIndex = float(avplIndex);
		buffer->depth = float(depth);
		buffer->id = float(id);
		buffer->intensity = intensity;
		buffer->mean = mean;
		buffer->normal = normal;
		buffer->size = float(size);
		buffer->pMin = bbox.pMin;
		buffer->pMax = bbox.pMax;
		buffer->materialIndex = materialIndex;
		buffer->incomingDirection = incomingDirection;

		if(left)
			buffer->left_id = float(left->id);
		else
			buffer->left_id = -1.f;
		if(right)
			buffer->right_id = float(right->id);
		else
			buffer->right_id = -1.f;
	}

	bool operator<(const CLUSTER& c) const {
		return (id < c.id);
	}

	bool operator<(const CLUSTER* c) const {
		return (id < c->id);
	}

	bool operator==(const CLUSTER& c) const {
		return (id == c.id);
	}

	bool operator==(const CLUSTER* c) const {
		return (id == c->id);
	}

	bool operator!=(const CLUSTER& c) const {
		return (id != c.id);
	}

	bool operator!=(const CLUSTER* c) const {
		return (id != c->id);
	}

	bool IsLeaf() const {
		return (left == 0 && right == 0);
	}

	float Distance(const CLUSTER* c1, const float weightNormals) const
	{
		if(c1 == 0) return std::numeric_limits<float>::max();

		//const float I = glm::length(intensity + c1->intensity);
		const BBox b = BBox::Union(bbox, c1->bbox);
		const float A = glm::length(b.pMax - b.pMin);
		//const float B = glm::dot(c1->normal, normal);

		const float dist = A; //I * ( A * A + weightNormals * weightNormals * (1 - B) * (1 - B));
		return dist;
	}
	
	BBox UpperBound(const CLUSTER* c)
	{
		BBox temp = BBox::Union(bbox, c->bbox);
		const float alpha = glm::length(temp.pMax - temp.pMin);
		const glm::vec3 diag = bbox.pMax - bbox.pMin;
		const float u = sqrtf(alpha*alpha-(diag.y*diag.y)-(diag.z-diag.z));
		const float v = sqrtf(alpha*alpha-(diag.z*diag.z)-(diag.x-diag.x));
		const float w = sqrtf(alpha*alpha-(diag.x*diag.x)-(diag.y-diag.y));
		const glm::vec3 d(u,v,w);
		temp.pMin = bbox.pMax - d;
		temp.pMax = bbox.pMin + d;
		return temp;
	}
};

struct CLUSTER_PAIR
{
	CLUSTER_PAIR() {
		c1 = c2 = 0;
	}

	CLUSTER_PAIR(CLUSTER* c1_param, CLUSTER* c2_param, const float dist_param)
		: c1(c1_param), c2(c2_param), dist(dist_param) { }
		
	bool operator==(const CLUSTER_PAIR cp) const {
		return (c1->id == cp.c1->id && c2->id == cp.c2->id);
	}

	bool operator==(const CLUSTER_PAIR* cp) const {
		return (c1->id == cp->c1->id && c2->id == cp->c2->id);
	}

	bool operator<(const CLUSTER_PAIR &cp) const
	{
		return dist < cp.dist;	
	}
	
	bool operator<(const CLUSTER_PAIR* cp) const
	{
		return dist < cp->dist;	
	}

	bool operator>(const CLUSTER_PAIR &cp) const
	{
		return dist > cp.dist;	
	}
		
	bool operator>(const CLUSTER_PAIR* cp) const
	{
		return dist > cp->dist;	
	}

	CLUSTER* c1;
	CLUSTER* c2;
	float dist;
};

struct CLUSTER_PAIR_COMPARE
{
    bool operator()(const CLUSTER_PAIR* lhs, const CLUSTER_PAIR* rhs) const
    {
		return lhs->dist > rhs->dist;
    }
};

struct HASH_CLUSTER
{
    size_t operator()(const CLUSTER* c) const
    {
        std::hash<int> H;
        return H(c->id);
    }
};

struct EQ_CLUSTER
{
    bool operator()(const CLUSTER* c1, const CLUSTER* c2) const {
        return c1->id==c2->id;
    }
};
/*
bool SORT_X(CLUSTER* p1, CLUSTER* p2);
bool SORT_Y(CLUSTER* p1, CLUSTER* p2);
bool SORT_Z(CLUSTER* p1, CLUSTER* p2);
*/
/*
bool SORT_X(const CLUSTER& p1, const CLUSTER& p2);
bool SORT_Y(const CLUSTER& p1, const CLUSTER& p2);
bool SORT_Z(const CLUSTER& p1, const CLUSTER& p2);
*/
#endif
