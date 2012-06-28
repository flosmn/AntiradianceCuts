#ifndef _BBOX_H_
#define _BBOX_H_

#include "Ray.h"

#include <glm/glm.hpp>

struct BBox
{
public:
	BBox(float min_x, float min_y, float min_z, float max_x, float max_y, float max_z)
	{
		pMin = glm::vec3(min_x, min_y, min_z);
		pMax = glm::vec3(max_x, max_y, max_z);
	}

	BBox(glm::vec3 min, glm::vec3 max)
	{
		pMin = min;
		pMax = max;
	}

	BBox() {}

	static BBox Union(const BBox& box0, const BBox& box1)
	{
		glm::vec3 min = glm::min(box0.pMin, box1.pMin);
		glm::vec3 max = glm::max(box0.pMax, box1.pMax);
		BBox res(min, max);
		return res;
	}

	float SurfaceArea() const {
		glm::vec3 d = pMax - pMin;
        return 2.f * (d.x * d.y + d.x * d.z + d.y * d.z);
    }

	// Returns the axis with the maximum extend (0 = x, 1=y, 2 = z axis)
	int MaximumExtent() const 
	{
        glm::vec3 diag = pMax - pMin;
        if (diag.x > diag.y && diag.x > diag.z)
            return 0;
        else if (diag.y > diag.z)
            return 1;
        else
            return 2;
    }

	bool IntersectP(const Ray &ray, float *hitt0, float *hitt1) const 
	{
		float t0 = ray.min_t, t1 = ray.max_t;
		for (int i = 0; i < 3; ++i) {
			// Update interval for i-th bounding box slab
			float invRayDir = 1.f / ray.d[i];
			float tNear = (pMin[i] - ray.o[i]) * invRayDir;
			float tFar  = (pMax[i] - ray.o[i]) * invRayDir;

			// Update parametric interval from slab intersection
			if (tNear > tFar)
			{ // swap
				float temp = tFar;
				tFar = tNear;
				tNear = temp;
			}
			t0 = tNear > t0 ? tNear : t0;
			t1 = tFar  < t1 ? tFar  : t1;
			if (t0 > t1) return false;
		}
		if (hitt0) *hitt0 = t0;
		if (hitt1) *hitt1 = t1;
		return true;
	}

	glm::vec3 pMin;
	glm::vec3 pMax;
	
};

#endif // _BBOX_H_