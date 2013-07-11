#ifndef _TRIANGLE_H_
#define _TRIANGLE_H_

#include <glm/glm.hpp>

#include "Intersection.h"
#include "BBox.h"
#include "Ray.h"
#include "Defines.h"

class Triangle
{
public:
	enum IsectMode { BACK_FACE, FRONT_FACE };

	Triangle() { }
	
	Triangle(glm::vec3 const& p0, glm::vec3 const& p1, glm::vec3 const& p2)
		: m_p0(p0), m_p1(p1), m_p2(p2) 
	{ 
		calcNormal();
		calcBBox();
	}

	Triangle(glm::vec3 const& p0, glm::vec3 const& p1, glm::vec3 const& p2, glm::vec3 const& n)
		: m_p0(p0), m_p1(p1), m_p2(p2), m_normal(n)
	{ 
		calcBBox();
	}

	bool intersectBBox(Ray const& ray) const 
	{
		const Ray r = ray;
		const glm::vec3 rayDirInv = 1.0f / r.getDirection();

		const glm::vec3 slabMin = (m_bbox.getMin() - r.getOrigin()) * rayDirInv;
		const glm::vec3 slabMax = (m_bbox.getMax() - r.getOrigin()) * rayDirInv;

		const glm::vec3 absMin = glm::min(slabMin, slabMax);
		const glm::vec3 absMax = glm::max(slabMin, slabMax);
			
		const float t0 = glm::max(0.0f, glm::max(absMin.x, glm::max(absMin.y, absMin.z)));
		const float t1 = glm::min(absMax.x, glm::min(absMax.y, absMax.z));

		bool intersection = t0 <= t1;
		return intersection;
	}

	bool intersect(const Ray& ray, float *dist, Intersection* intersection, IsectMode isectMode) const
	{
		const glm::vec3 e1 = (isectMode == FRONT_FACE) ? m_p2 - m_p0 : m_p1 - m_p0; 
		const glm::vec3 e2 = (isectMode == FRONT_FACE) ? m_p1 - m_p0 : m_p2 - m_p0; 

		const glm::vec3 p = glm::cross(ray.getDirection(), e2);
		const float det = glm::dot(e1, p);

		if(det < EPS&& det > -EPS) {
			return false;
		}
		
		const glm::vec3 t = ray.getOrigin() - m_p0;
		const float u = glm::dot(t, p);

		if(u < 0.0f || u > det) {
			return false;
		}

		const glm::vec3 q = glm::cross(t, e1);
		const float v = glm::dot(ray.getDirection(), q);

		if(v < 0.0f || u + v > det) {
			return false;
		}

		*dist = 1.f/det * glm::dot(e2, q);

		intersection->setPosition(ray.getOrigin() + (*dist) * ray.getDirection());
		intersection->setPrimitive(this);

		return true;
	}

	void transform(Triangle* triangle, glm::mat4 const& transform) const
	{
		glm::vec4 t_p0 = transform * glm::vec4(m_p0, 1.0f);
		glm::vec4 t_p1 = transform * glm::vec4(m_p1, 1.0f);
		glm::vec4 t_p2 = transform * glm::vec4(m_p2, 1.0f);
		
		t_p0 /= t_p0.w;
		t_p1 /= t_p1.w;
		t_p2 /= t_p2.w;
		
		triangle->setPoints(glm::vec3(t_p0), glm::vec3(t_p1), glm::vec3(t_p2));
	}
	
	inline BBox const& getBBox() const { return m_bbox; }
	inline glm::vec3 const& P0() const { return m_p0; }
	inline glm::vec3 const& P1() const { return m_p1; }
	inline glm::vec3 const& P2() const { return m_p2; }
	inline glm::vec3 const& getNormal() const { return m_normal; }

	inline void setMaterialIndex(int i) { m_materialIndex = i; }
	inline int getMaterialIndex() const { return m_materialIndex; }

private:
	void setPoints(glm::vec3 const& p0, glm::vec3 const& p1, glm::vec3 const& p2) {
		m_p0 = p0;
		m_p1 = p1;
		m_p2 = p2;

		calcBBox();
		calcNormal();
	}

	void calcNormal() {
		glm::vec3 temp = glm::cross(m_p2 - m_p0, m_p1 - m_p0);
		if (glm::length(temp) > 0.f) temp = glm::normalize(temp);
		m_normal = temp;
	}

	void calcBBox() {
		m_bbox = BBox(glm::min(glm::min(m_p0, m_p1), m_p2),
			glm::max(glm::max(m_p0, m_p1), m_p2));
	}

private:
	int m_materialIndex;
	glm::vec3 m_p0;
	glm::vec3 m_p1;
	glm::vec3 m_p2;
	glm::vec3 m_normal;
	BBox m_bbox;
};

#endif // _TRIANGLE_H_
