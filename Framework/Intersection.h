#ifndef INTERSECTION_H
#define INTERSECTION_H

#include "glm/glm.hpp"

class Triangle;

class Intersection
{
public:
	Intersection() : m_position(glm::vec3(0.f)), m_triangle(nullptr)
	{ }

	Intersection(glm::vec3 const& position, Triangle const* triangle) : 
		m_position(position),
		m_triangle(triangle)
	{ }
	
	~Intersection() {}	

	inline Triangle const* getTriangle() const { 
		return m_triangle; 
	}
	
	inline glm::vec3 const& getPosition() const { 
		return m_position; 
	}

	inline void setPrimitive(Triangle const* const triangle) {
		m_triangle = triangle; 
	}
	
	inline void setPosition(const glm::vec3& position) { 
		m_position = position; 
	}

private:
	Triangle const* m_triangle;
	glm::vec3 m_position;
};

#endif