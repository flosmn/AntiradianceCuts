#ifndef _BBOX_H_
#define _BBOX_H_

#include "Ray.h"

#include <glm/glm.hpp>

#include <sstream>

class BBox
{
public:
	BBox(float min_x, float min_y, float min_z, 
		 float max_x, float max_y, float max_z) :
		m_min(glm::vec3(min_x, min_y, min_z)),
		m_max(glm::vec3(max_x, max_y, max_z))
	{ }

	BBox(glm::vec3 const& min, glm::vec3 const& max) :
		m_min(min), m_max(max)
	{ }

	BBox() : 
		m_max(glm::vec3(std::numeric_limits<float>::min())),
		m_min(glm::vec3(std::numeric_limits<float>::max()))
	{ }

	inline static BBox Union(BBox const& box0, BBox const& box1) {
		return BBox(glm::min(box0.getMin(), box1.getMin()),
			glm::max(box0.getMax(), box1.getMax()));
	}

	inline float getDistance(glm::vec3 const& p) const {
		glm::vec3 cp; // closest point on bbox to p
		cp.x = (p.x < m_min.x) ? m_min.x : (p.x > m_max.x) ? m_max.x : p.x;
		cp.y = (p.y < m_min.y) ? m_min.y : (p.y > m_max.y) ? m_max.y : p.y;
		cp.z = (p.z < m_min.z) ? m_min.z : (p.z > m_max.z) ? m_max.z : p.z;
		return glm::length(p - cp);
	}

	inline float getSurfaceArea() const {
		const glm::vec3 d = m_max - m_min;
        return 2.f * (d.x * d.y + d.x * d.z + d.y * d.z);
    }

	inline bool intersects(BBox const& other) const {
      return 
        (m_min.x < other.m_max.x) && (m_max.x > other.m_min.x) &&
        (m_min.y < other.m_max.y) && (m_max.y > other.m_min.y) &&
        (m_min.z < other.m_max.z) && (m_max.z > other.m_min.z);
    }

	// Returns the axis with the maximum extend (0 = x, 1=y, 2 = z axis)
	inline int getAxisMaximumExtent() const {
        const glm::vec3 diag = m_max - m_min;
        if (diag.x > diag.y && diag.x > diag.z)
            return 0;
        else if (diag.y > diag.z)
            return 1;
        else
            return 2;
    }

	inline bool intersect(Ray const& ray, float *t_min, float *t_max) const 
	{
		assert(t_min != nullptr);
		assert(t_max != nullptr);
		
		float t0 = ray.getMin(), t1 = ray.getMax();
		for (int i = 0; i < 3; ++i) {
			// Update interval for i-th bounding box slab
			const float invRayDir = 1.f / ray.getDirection()[i];
			float tNear = (m_min[i] - ray.getOrigin()[i]) * invRayDir;
			float tFar  = (m_max[i] - ray.getOrigin()[i]) * invRayDir;

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
		*t_min = t0;
		*t_max = t1;
		return true;
	}

	inline glm::vec3 getMin() const {
		return m_min; 
	}
	
	inline glm::vec3 getMax() const {
		return m_max; 
	}
	
	inline void setMin(glm::vec3 const& min) {
		m_min = min; 
	}
	
	inline void setMax(glm::vec3 const& max) {
		m_max = max; 
	}

private:
	glm::vec3 m_min;
	glm::vec3 m_max;
};

#endif // _BBOX_H_
