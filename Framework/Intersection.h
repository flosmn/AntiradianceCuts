#ifndef INTERSECTION_H
#define INTERSECTION_H

#include "glm/glm.hpp"

#include "CTriangle.h"

class CModel;
class CPrimitive;

class Intersection
{
public:
	Intersection() { m_Position = glm::vec3(0.f); m_pPrimitive = 0; }
	Intersection(const glm::vec3& position, CPrimitive* pPrimitive) { m_Position = position; m_pPrimitive = pPrimitive; }
	~Intersection() {}	

	CPrimitive* GetPrimitive() const { return m_pPrimitive; }
	glm::vec3 GetPosition() const { return m_Position; }
	
	glm::vec3 Intersection::GetNormal() const;
	uint GetMaterialIndex () const;

	void SetPrimitive(CPrimitive* pPrimitive) { m_pPrimitive = pPrimitive; }
	void SetPosition(const glm::vec3& position) { m_Position = position; }

private:
	CPrimitive* m_pPrimitive;
	glm::vec3 m_Position;
};

struct SceneSample
{
	SceneSample() {}
	SceneSample(const Intersection& i)
		: position(i.GetPosition()), normal(i.GetNormal()), materialIndex(i.GetMaterialIndex())
	{}

	glm::vec3 position;
	glm::vec3 normal;
	uint materialIndex;
	float pdf;
};

#endif