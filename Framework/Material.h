#ifndef _MATERIAL_H_
#define _MATERIAL_H_

#include <glm/glm.hpp>

struct MATERIAL
{
	glm::vec3 emissive;
	glm::vec3 diffuse;
	glm::vec3 specular;
	float exponent;
};

#endif // _MATERIAL_H_
