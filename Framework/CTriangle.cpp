#include "CTriangle.h"

#include <algorithm>

CTriangle::CTriangle(glm::vec3 p0, glm::vec3 p1, glm::vec3 p2)
	: m_P0(p0), m_P1(p1), m_P2(p2)
{
	glm::vec3 min = glm::min(glm::min(m_P0, m_P1), m_P2);
	glm::vec3 max = glm::max(glm::max(m_P0, m_P1), m_P2);

	BBox bbox(min, max);

	m_BBox = bbox;
}

CTriangle::~CTriangle()
{

}

bool CTriangle::IntersectBBox(const Ray& ray)
{
	Ray r = ray;

	glm::vec3 rayDirInv = 1.0f / r.d;

	glm::vec3 slabMin = (m_BBox.pMin - r.o) * rayDirInv;
	glm::vec3 slabMax = (m_BBox.pMax - r.o) * rayDirInv;

	glm::vec3 absMin = glm::min(slabMin, slabMax);
	glm::vec3 absMax = glm::max(slabMin, slabMax);
		
	float t0 = glm::max(0.0f, glm::max(absMin.x, glm::max(absMin.y, absMin.z)));
	float t1 = glm::min(absMax.x, glm::min(absMax.y, absMax.z));

	bool intersection = t0 <= t1;
	return intersection;
}

bool CTriangle::Intersect(const Ray& ray, float *t, IntersectionNew* pIntersection)
{
	Ray r = ray;
	float epsilon = 0.0001f;

	glm::vec3 vec_p, vec_t, vec_q;
	glm::vec3 e1 = m_P1 - m_P0;
	glm::vec3 e2 = m_P2 - m_P0;

	vec_p = glm::cross(r.d, e2);

	float det = glm::dot(e1, vec_p);

	if(det < epsilon && det > -epsilon) return false;
	
	vec_t = r.o - m_P0;

	float u = glm::dot(vec_t, vec_p);

	if(u < 0.0f || u > det) return false;

	vec_q = glm::cross(vec_t, e1);

	float v = glm::dot(r.d, vec_q);

	if(v < 0.0f || u + v > det) return false;

	*t = glm::dot(e2, vec_q);
	*t *= 1.0f/det;

	pIntersection->position = ray.o + (*t) * ray.d;
	//pIntersection->pPrimitive = this;

	return true;
}

BBox CTriangle::GetBBox()
{
	return m_BBox;
}