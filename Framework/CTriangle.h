#ifndef _C_TRIANGLE_H_
#define _C_TRIANGLE_H_

#include <glm/glm.hpp>

#include "CPrimitive.h"

#include "BBox.h"
#include "Ray.h"

class Intersection;

class CTriangle : public CPrimitive
{
public:
	CTriangle() {};
	CTriangle(glm::vec3 p0, glm::vec3 p1, glm::vec3 p2);
	CTriangle(glm::vec3 p0, glm::vec3 p1, glm::vec3 p2, glm::vec3 n);
	~CTriangle();

	virtual bool IntersectBBox(const Ray& ray);
	virtual bool Intersect(const Ray& ray, float *t, Intersection* pIntersection);
	virtual BBox GetBBox();
	virtual void Transform(CPrimitive* pPrimitive, const glm::mat4& transform) const;
		
	void SetPoints(glm::vec3 p0, glm::vec3 p1, glm::vec3 p2);
	
	glm::vec3 P0() const { return m_P0; }
	glm::vec3 P1() const { return m_P1; }
	glm::vec3 P2() const { return m_P2; }
	glm::vec3 GetNormal() const { return m_Normal; }

private:
	void CalcNormal();
	void CalcBBox();

	glm::vec3 m_P0;
	glm::vec3 m_P1;
	glm::vec3 m_P2;
	glm::vec3 m_Normal;
	BBox m_BBox;
};

#endif // _C_TRIANGLE_H_