
#ifndef _OCTAHEDRON_UTIL_H_
#define _OCTAHEDRON_UTIL_H_

#include <glm/glm.hpp>

float sign(const float& f)
{
	float res;
	res = f > 0 ? 1.f : -1.f;
	return res;
}

glm::vec2 sign(const glm::vec2& v)
{
	glm::vec2 res;
	res.x = v.x > 0 ? 1.f : -1.f;
	res.y = v.y > 0 ? 1.f : -1.f;
	return res;
}

glm::vec3 sign(const glm::vec3& v)
{
	glm::vec3 res;
	res.x = v.x > 0 ? 1.f : -1.f;
	res.y = v.y > 0 ? 1.f : -1.f;
	res.z = v.z > 0 ? 1.f : -1.f;
	return res;
}

glm::vec2 abs(const glm::vec2& v)
{
	glm::vec2 res;
	res.x = v.x > 0 ? v.x : -v.x;
	res.y = v.y > 0 ? v.y : -v.y;
	return res;
}

glm::vec3 abs(const glm::vec3& v)
{
	glm::vec3 res;
	res.x = v.x > 0 ? v.x : -v.x;
	res.y = v.y > 0 ? v.y : -v.y;
	res.z = v.z > 0 ? v.z : -v.z;
	return res;
}

#endif _OCTAHEDRON_UTIL_H_