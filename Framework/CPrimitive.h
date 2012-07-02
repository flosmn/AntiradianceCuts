#ifndef _C_PRIMITIVE_H_
#define _C_PRIMITIVE_H_

#include <glm/glm.hpp>

#include "BBox.h"
#include "Ray.h"

class Intersection;
class CModel;

class CPrimitive
{
public:
	CPrimitive() { }
	CPrimitive(CModel* pModel) { m_pModel = pModel; }
	virtual ~CPrimitive() { }

	virtual bool IntersectBBox(const Ray& ray) = 0;
	virtual bool Intersect(const Ray& ray, float* t, Intersection* pIntersection) = 0;
	virtual BBox GetBBox() = 0;
	virtual glm::vec3 GetNormal() const = 0;
	virtual void Transform(CPrimitive* pPrimitive, const glm::mat4& transform) const = 0;
	
	CModel* GetModel() const { return m_pModel; }
	void SetModel(CModel* pModel) { m_pModel = pModel; }

private:
	CModel* m_pModel;
};

#endif _C_PRIMITIVE_H_