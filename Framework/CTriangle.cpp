#include "CTriangle.h"

#include "MeshResources\CModel.h"

#include "Intersection.h"

#include <algorithm>

CTriangle::CTriangle(glm::vec3 p0, glm::vec3 p1, glm::vec3 p2)
	: m_P0(p0), m_P1(p1), m_P2(p2)
{
	CalcBBox();
	CalcNormal();
}

CTriangle::CTriangle(glm::vec3 p0, glm::vec3 p1, glm::vec3 p2, glm::vec3 n)
	: m_P0(p0), m_P1(p1), m_P2(p2), m_Normal(n)
{
	CalcBBox();	
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

bool CTriangle::Intersect(const Ray& ray, float *t, Intersection* pIntersection, bool back_face_culling)
{
	if(!back_face_culling)
	{
		bool isect_bf = IntersectBackFace(ray, t, pIntersection);
		
		if(isect_bf) 
			return true;
	}
	
	float epsilon = 0.0001f;

	glm::vec3 vec_p, vec_t, vec_q;
	glm::vec3 e1 = m_P2 - m_P0;
	glm::vec3 e2 = m_P1 - m_P0;

	vec_p = glm::cross(ray.d, e2);

	float det = glm::dot(e1, vec_p);

	if(det < epsilon && det > -epsilon) return false;
	
	vec_t = ray.o - m_P0;

	float u = glm::dot(vec_t, vec_p);

	if(u < 0.0f || u > det) return false;

	vec_q = glm::cross(vec_t, e1);

	float v = glm::dot(ray.d, vec_q);

	if(v < 0.0f || u + v > det) return false;

	*t = glm::dot(e2, vec_q);
	*t *= 1.0f/det;

	pIntersection->SetPosition(ray.o + *t * ray.d);
	pIntersection->SetPrimitive(this);

	return true;
}

bool CTriangle::IntersectBackFace(const Ray& ray, float *t, Intersection* pIntersection)
{
	float epsilon = 0.0001f;

	glm::vec3 vec_p, vec_t, vec_q;
	glm::vec3 e1 = m_P1 - m_P0;
	glm::vec3 e2 = m_P2 - m_P0;

	vec_p = glm::cross(ray.d, e2);

	float det = glm::dot(e1, vec_p);

	if(det < epsilon && det > -epsilon) return false;
	
	vec_t = ray.o - m_P0;

	float u = glm::dot(vec_t, vec_p);

	if(u < 0.0f || u > det) return false;

	vec_q = glm::cross(vec_t, e1);

	float v = glm::dot(ray.d, vec_q);

	if(v < 0.0f || u + v > det) return false;

	*t = glm::dot(e2, vec_q);
	*t *= 1.0f/det;

	pIntersection->SetPosition(ray.o + *t * ray.d);
	pIntersection->SetPrimitive(this);

	return true;
}

BBox CTriangle::GetBBox()
{
	return m_BBox;
}

void CTriangle::Transform(CPrimitive* pPrimitive, const glm::mat4& transform) const
{
	glm::vec4 t_p0 = transform * glm::vec4(m_P0, 1.0f);
	glm::vec4 t_p1 = transform * glm::vec4(m_P1, 1.0f);
	glm::vec4 t_p2 = transform * glm::vec4(m_P2, 1.0f);
	
	t_p0 /= t_p1.w;
	t_p1 /= t_p2.w;
	t_p2 /= t_p2.w;
		
	glm::vec3 t2_p0 = glm::vec3(t_p0);
	glm::vec3 t2_p1 = glm::vec3(t_p1);
	glm::vec3 t2_p2 = glm::vec3(t_p2);
	
	CTriangle* t = (CTriangle*)pPrimitive;
	t->SetPoints(t2_p0, t2_p1, t2_p2);
}

void CTriangle::SetPoints(glm::vec3 p0, glm::vec3 p1, glm::vec3 p2)
{
	m_P0 = p0;
	m_P1 = p1;
	m_P2 = p2;

	CalcNormal();
	CalcBBox();
}

void CTriangle::CalcNormal()
{
	m_Normal = glm::normalize(glm::cross(m_P2 - m_P0, m_P1 - m_P0));
}

void CTriangle::CalcBBox()
{
	glm::vec3 min = glm::min(glm::min(m_P0, m_P1), m_P2);
	glm::vec3 max = glm::max(glm::max(m_P0, m_P1), m_P2);

	BBox bbox(min, max);

	m_BBox = bbox;
}