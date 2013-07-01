#ifndef RAY_H
#define RAY_H

#include "glm/glm.hpp"

#include <limits>

class Ray
{
public:
	Ray(glm::vec3 const& origin, glm::vec3 const& direction) :
		m_origin(origin), m_direction(direction),
		m_min(std::numeric_limits<float>::min()),
		m_max(std::numeric_limits<float>::max())
	{
	} 

	inline glm::vec3 const& getOrigin() const { 
		return m_origin; 
	}

	inline glm::vec3 const& getDirection() const { 
		return m_direction; 
	}
	
	inline float getMax() const { 
		return m_max; 
	}
	
	inline float getMin() const { 
		return m_min; 
	}
	
	inline void setOrigin(glm::vec3 const& origin) { 
		m_origin = origin; 
	}
	
	inline void setDirection(glm::vec3 const& direction) { 
		m_direction = direction; 
	}
	
	inline void setMin(float min) { 
		m_min = min; 
	}
	
	inline void setMax(float max) { 
		m_max = max; 
	}

private:
	glm::vec3 m_origin;
	glm::vec3 m_direction;

	// to restrict the ray
	float m_min;
	float m_max;
};

#endif
