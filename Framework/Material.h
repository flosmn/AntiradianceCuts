#ifndef _MATERIAL_H_
#define _MATERIAL_H_

#include <glm/glm.hpp>

struct MATERIAL
{
	glm::vec4 emissive;
	glm::vec4 diffuse;
	glm::vec4 specular;
	float exponent;
	float padd0;
	float padd1;
	float padd2;
};

#endif // _MATERIAL_H_
