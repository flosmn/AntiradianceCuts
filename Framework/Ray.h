#ifndef RAY_H
#define RAY_H

#include "glm/glm.hpp"

class Ray
{
public:
	Ray(glm::vec3 _origin, glm::vec3 _direction) 
	{
			origin = _origin;
			direction = _direction;
	}

	glm::vec3 GetOrigin() { return origin; }
	glm::vec3 GetDirection() { return direction; }

	void SetOrigin(glm::vec3 _origin) { origin = _origin; }

private:
	glm::vec3 origin;
	glm::vec3 direction;
};

#endif