#ifndef _C_TRIANGLE_H_
#define _C_TRIANGLE_H_

#include <glm/glm.hpp>

#include "CPrimitive.h"

#include "BBox.h"
#include "Ray.h"
#include "IntersectionNew.h"

class CTriangle : public CPrimitive
{
public:
	CTriangle(glm::vec3 p0, glm::vec3 p1, glm::vec3 p2);
	~CTriangle();

	virtual bool IntersectBBox(const Ray& ray);
	virtual bool Intersect(const Ray& ray, float *t, IntersectionNew* pIntersection);
	virtual BBox GetBBox();
	
	glm::vec3 P0() { return m_P0; }
	glm::vec3 P1() { return m_P1; }
	glm::vec3 P2() { return m_P2; }

private:
	glm::vec3 m_P0;
	glm::vec3 m_P1;
	glm::vec3 m_P2;
	BBox m_BBox;
};

#endif // _C_TRIANGLE_H_