#ifndef _C_PRIMITIVE_H_
#define _C_PRIMITIVE_H_

#include "BBox.h"
#include "Ray.h"
#include "IntersectionNew.h"

class CPrimitive
{
public:
	CPrimitive() { };
	virtual ~CPrimitive() { };

	virtual bool IntersectBBox(const Ray& ray) = 0;
	virtual bool Intersect(const Ray& ray, float* t, IntersectionNew* pIntersection) = 0;
	virtual BBox GetBBox() = 0;
};

#endif _C_PRIMITIVE_H_