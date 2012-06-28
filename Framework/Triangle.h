#ifndef TRIANGLE_H
#define TRIANGLE_H

#include "glm/glm.hpp"

class Triangle
{
public:
	Triangle() {};
	
	Triangle(glm::vec3 p0_param, glm::vec3 p1_param, glm::vec3 p2_param, glm::vec3 n_param)
	{
		n = n_param;
		p0 = p0_param;
		p1 = p1_param;
		p2 = p2_param;
	}

	Triangle GetTransformedTriangle(glm::mat4 transform)
	{
		glm::vec4 t_p0 = transform * glm::vec4(p0, 1.0f);
		glm::vec4 t_p1 = transform * glm::vec4(p1, 1.0f);
		glm::vec4 t_p2 = transform * glm::vec4(p2, 1.0f);
		t_p0 /= t_p1.w;
		t_p1 /= t_p2.w;
		t_p2 /= t_p2.w;
		
		glm::vec3 t2_p0 = glm::vec3(t_p0);
		glm::vec3 t2_p1 = glm::vec3(t_p1);
		glm::vec3 t2_p2 = glm::vec3(t_p2);

		glm::vec3 normal = glm::normalize(glm::cross(t2_p2-t2_p0, t2_p1-t2_p0));

		return Triangle(t2_p0, t2_p1, t2_p2, normal);
	}

	glm::vec3 n;
	glm::vec3 p0;
	glm::vec3 p1;
	glm::vec3 p2;
};

#endif