#ifndef INTERSECTION_H
#define INTERSECTION_H

#include "glm/glm.hpp"

#include "Triangle.h"

class CModel;

class Intersection
{
public:
	Intersection() 
	{
		valid = false;
	}
	
	Intersection(CModel* _model, Triangle _triangle, glm::vec3 _point)
	{
		model = _model;
		triangle = _triangle;
		point = _point;
		valid = true;
	}

	CModel* GetModel() { return model; }
	Triangle GetTriangle() { return triangle; }
	glm::vec3 GetPoint() { return point; }

	bool IsValid() { return valid; } 

private:
	bool valid;
	CModel* model;
	Triangle triangle;
	glm::vec3 point;
};

#endif