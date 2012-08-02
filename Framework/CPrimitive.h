#ifndef _C_PRIMITIVE_H_
#define _C_PRIMITIVE_H_

#include <glm/glm.hpp>

#include "Structs.h"

#include "BBox.h"
#include "Ray.h"

class Intersection;

class CPrimitive
{
public:
	CPrimitive() { }
	virtual ~CPrimitive() { }

	virtual bool IntersectBBox(const Ray& ray) = 0;
	virtual bool Intersect(const Ray& ray, float* t, Intersection* pIntersection, bool back_face_culling) = 0;
	virtual BBox GetBBox() = 0;
	virtual glm::vec3 GetNormal() const = 0;
	virtual void Transform(CPrimitive* pPrimitive, const glm::mat4& transform) const = 0;
	
	const MATERIAL GetMaterial() const { return m_Material; }
	void SetMaterial(const MATERIAL& mat) { m_Material = mat; }

private:
	MATERIAL m_Material;
};

#endif _C_PRIMITIVE_H_