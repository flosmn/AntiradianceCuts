#ifndef INTERSECTION_H
#define INTERSECTION_H

#include "glm/glm.hpp"

#include "CTriangle.h"

class CModel;

class Intersection
{
public:
	Intersection() { m_Position = glm::vec3(0.f); m_pPrimitive = 0; }
	Intersection(const glm::vec3& position, CPrimitive* pPrimitive) { m_Position = position; m_pPrimitive = pPrimitive; }
	~Intersection() {}	

	CPrimitive* GetPrimitive() const { return m_pPrimitive; }
	glm::vec3 GetPosition() const { return m_Position; }
	glm::vec3 GetNormal() const { return m_pPrimitive->GetNormal(); }
	MATERIAL GetMaterial() const { return m_pPrimitive->GetMaterial(); }

	void SetPrimitive(CPrimitive* pPrimitive) { m_pPrimitive = pPrimitive; }
	void SetPosition(const glm::vec3& position) { m_Position = position; }

private:
	CPrimitive* m_pPrimitive;
	glm::vec3 m_Position;
};

#endif