#ifndef _C_PRIMITIVE_H_
#define _C_PRIMITIVE_H_

struct BBox
{
	BBox(float min_x, float min_y, float min_z, float max_x, float max_y, float max_z)
	{
		m_Min = glm::vec3(min_x, min_y, min_z);
		m_Max = glm::vec3(max_x, max_y, max_z);
	}

	BBox(glm::vec3 min, glm::vec3 max)
	{
		m_Min = min;
		m_Max = max;
	}

	float MinX() { return m_Min.x; }
	float MinY() { return m_Min.y; }
	float MinZ() { return m_Min.z; }

	float MaxX() { return m_Max.x; }
	float MaxY() { return m_Max.y; }
	float MaxZ() { return m_Max.z; }

	glm::vec3 m_Min;
	glm::vec3 m_Max;
};

class CPrimitive
{
public:
	CPrimitive();
	~CPrimitive();

	virtual void IntersectBBox(const Ray& ray) = 0;
	virtual void Intersect(const Ray& ray) = 0;
	virtual BBox GetBBox() = 0;
	virtual float GetDistance() = 0;
};

class CTriangle : public CPrimitive
{
	CTriangle(glm::vec3 p0, glm::vec3 p1, glm::vec3 p2);
	~CTriangle();

	virtual void IntersectBBox(const Ray& ray);
	virtual void Intersect(const Ray& ray);
	virtual BBox GetBBox();

	glm::vec3 P0() { return m_P0; }
	glm::vec3 P1() { return m_P1; }
	glm::vec3 P2() { return m_P2; }

private:
	glm::vec3 m_P0;
	glm::vec3 m_P1;
	glm::vec3 m_P2;
	BBox m_BBox;
}

#endif _C_PRIMITIVE_H_