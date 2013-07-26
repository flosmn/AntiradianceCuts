#ifndef UTIL_H_
#define UTIL_H_

#include <glm/glm.hpp>

#include <iostream>

static inline glm::mat3 tangentSpace(glm::vec3 const& n) {
	// never co-linear
	const glm::vec3 t(n.y, n.z, -n.x);
	const glm::vec3 U = glm::normalize(glm::cross(t, n));
	const glm::vec3 V = glm::normalize(glm::cross(U, n));
	return glm::mat3(U, V, n);
}

static inline float luminance(glm::vec3 const& v) {
	return 0.2126f * v.r + 0.7152f * v.g + 0.0722f * v.b;
}

static inline glm::vec3 reflect(glm::vec3 const& v, glm::vec3 const& n)
{
	const float cos_theta = glm::dot(v,n);
#ifdef _DEBUG
	if(cos_theta < 0.f)
		std::cout << "reflect wrong" << std::endl;
#endif
	return glm::normalize(2.f * cos_theta * n - v);
}

#endif // UTIL_H_