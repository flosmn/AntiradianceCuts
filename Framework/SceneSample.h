#ifndef SCENESAMPLE_H_
#define SCENESAMPLE_H_

#include "Triangle.h"

struct SceneSample
{
	SceneSample() {}
	SceneSample(Intersection const& i) :
		position(i.getPosition()),
		normal(i.getTriangle()->getNormal()), 
		materialIndex(i.getTriangle()->getMaterialIndex())
	{}

	glm::vec3 position;
	glm::vec3 normal;
	uint materialIndex;
	float pdf;
};


#endif SCENESAMPLE_H_
