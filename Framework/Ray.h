#ifndef RAY_H
#define RAY_H

#include "glm/glm.hpp"

#include <limits>

class Ray
{
public:
	Ray(glm::vec3 origin, glm::vec3 direction)
	{
		o = origin;
		d = direction;
		min_t = std::numeric_limits<float>::min();
		max_t = std::numeric_limits<float>::max();
	}

	glm::vec3 o;
	glm::vec3 d;

	// to restrict the ray
	float min_t;
	float max_t;

	void SetOrigin(glm::vec3 origin) { o = origin; }	
};

#endif