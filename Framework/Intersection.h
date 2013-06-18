#ifndef INTERSECTION_H
#define INTERSECTION_H

typedef unsigned int uint;

#include "glm/glm.hpp"

class CTriangle;
class CModel;

class Intersection
{
public:
	Intersection() { m_Position = glm::vec3(0.f); m_pPrimitive = 0; }
	Intersection(const glm::vec3& position, CTriangle* pPrimitive) { m_Position = position; m_pPrimitive = pPrimitive; }
	~Intersection() {}	

	CTriangle const* GetPrimitive() const { return m_pPrimitive; }
	glm::vec3 GetPosition() const { return m_Position; }
	
	glm::vec3 GetNormal() const;
	uint GetMaterialIndex () const;

	void SetPrimitive(CTriangle const* const pPrimitive) { m_pPrimitive = pPrimitive; }
	void SetPosition(const glm::vec3& position) { m_Position = position; }

private:
	CTriangle const* m_pPrimitive;
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