#ifndef _C_TRIANGLE_H_
#define _C_TRIANGLE_H_

#include <glm/glm.hpp>

#include "Intersection.h"
#include "BBox.h"
#include "Ray.h"

class CTriangle
{
public:
	enum IsectMode { BACK_FACE, FRONT_FACE };

	CTriangle() {};
	CTriangle(glm::vec3 p0, glm::vec3 p1, glm::vec3 p2);
	CTriangle(glm::vec3 p0, glm::vec3 p1, glm::vec3 p2, glm::vec3 n);
	~CTriangle();

	virtual bool IntersectBBox(const Ray& ray) const;
	virtual bool Intersect(const Ray& ray, float *t, Intersection* pIntersection, IsectMode isectMode) const;
	virtual BBox GetBBox();
	virtual void Transform(CTriangle* pPrimitive, const glm::mat4& transform) const;
		
	void SetPoints(glm::vec3 p0, glm::vec3 p1, glm::vec3 p2);
	
	glm::vec3 P0() const { return m_P0; }
	glm::vec3 P1() const { return m_P1; }
	glm::vec3 P2() const { return m_P2; }
	glm::vec3 GetNormal() const { return m_Normal; }

	void SetMaterialIndex(uint i) { m_MaterialIndex = i; }
	uint GetMaterialIndex() const { return m_MaterialIndex; }

private:
	void CalcNormal();
	void CalcBBox();

	uint m_MaterialIndex;
	glm::vec3 m_P0;
	glm::vec3 m_P1;
	glm::vec3 m_P2;
	glm::vec3 m_Normal;
	BBox m_BBox;
};

#endif // _C_TRIANGLE_H_
