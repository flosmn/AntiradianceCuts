#ifndef _INTERSECTION_NEW_H_
#define _INTERSECTION_NEW_H_

#include <glm/glm.hpp>

#include "CPrimitive.h"

class IntersectionNew
{
public:
	IntersectionNew() {}
	//IntersectionNew(const CPrimitive& p, glm::vec3 pos) { pPrimitive = p; position = pos; }

	//CPrimitive* pPrimitive;
	glm::vec3 position;
};

#endif _INTERSECTION_NEW_H_