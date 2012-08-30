#ifndef _MATERIAL_H_
#define _MATERIAL_H_

#include <glm/glm.hpp>

struct MATERIAL
{
	MATERIAL() : 
		emissive(glm::vec4(0.f)),
		diffuse(glm::vec4(0.f)),
		specular(glm::vec4(0.f)), 
		exponent(0.f),
		padd0(0.f),
		padd1(0.f),
		padd2(0.f) {}

	MATERIAL(glm::vec4 e, glm::vec4 d, glm::vec4 s, float exp) : 
		emissive(e),
		diffuse(d),
		specular(s), 
		exponent(exp),
		padd0(0.f),
		padd1(0.f),
		padd2(0.f) {}

	glm::vec4 emissive;
	glm::vec4 diffuse;
	glm::vec4 specular;
	float exponent;
	float padd0;
	float padd1;
	float padd2;
};

#endif _MATERIAL_H_