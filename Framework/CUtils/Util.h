#ifndef UTIL_H
#define UTIL_H

typedef unsigned short ushort;
typedef unsigned int uint;

#include "glm/glm.hpp"

#include "..\Defines.h"
#include "..\Triangle.h"
#include "..\Ray.h"

#include "..\rand.h"

#include <vector>
#include <string>

glm::mat4 IdentityMatrix();

float map(float x, float x0, float x1, float y0, float y1);

float Rad2Deg (float Angle);

glm::vec2 GetUniformRandomSample2D(glm::vec2 range_u, glm::vec2 range_v);
glm::vec3 GetRandomSampleDirectionCosCone(glm::vec3 orientation, float&pdf, uint order);
glm::vec3 GetRandomSampleDirectionCosCone(glm::vec3 orientation, const float u1, const float u2, float&pdf, uint order);

float map(float x, float x0, float x1, float y0, float y1);

bool IntersectRayBox(glm::vec3 rayPos, glm::vec3 rayDir, glm::vec3 boxMin, 
										 glm::vec3 boxMax);

bool IntersectWithBB(Triangle triangle, Ray ray);

bool IntersectRayTriangle(glm::vec3 origin, glm::vec3 direction, 
							 glm::vec3 v0, glm::vec3 v1, glm::vec3 v2, float &t);

std::string AsString(glm::vec3 v);
std::string AsString(glm::vec4 v);

void PlaneHammersley(float *result, int n);

#endif