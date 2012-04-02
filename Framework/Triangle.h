#ifndef TRIANGLE_H
#define TRIANGLE_H

#include "glm/glm.hpp"

class Triangle
{
public:
	Triangle() {};
	
	Triangle(glm::vec3 p1, glm::vec3 p2, glm::vec3 p3, glm::vec3 n)
	{
		normal = n;
		points[0] = p1;
		points[1] = p2;
		points[2] = p3;
	}

	glm::vec3 GetNormal() { return normal; }
	glm::vec3* GetPoints() { return points; }

	Triangle GetTransformedTriangle(glm::mat4 transform)
	{
		glm::vec4 t_p1 = transform * glm::vec4(points[0], 1.0f);
		glm::vec4 t_p2 = transform * glm::vec4(points[1], 1.0f);
		glm::vec4 t_p3 = transform * glm::vec4(points[2], 1.0f);
		t_p1 /= t_p1.w;
		t_p2 /= t_p2.w;
		t_p2 /= t_p2.w;
		
		glm::vec3 p1 = glm::vec3(t_p1);
		glm::vec3 p2 = glm::vec3(t_p2);
		glm::vec3 p3 = glm::vec3(t_p3);

		glm::vec3 normal = glm::normalize(glm::cross(p3-p1, p2-p1));

		return Triangle(p1, p2, p3, normal);
	}

private:
	glm::vec3 normal;
	glm::vec3 points[3];
};

#endif