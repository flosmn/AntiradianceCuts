#ifndef UTIL_H
#define UTIL_H

typedef unsigned int uint;

#include "glm/glm.hpp"

#include "..\Defines.h"
#include "..\CTriangle.h"
#include "..\Ray.h"
#include "..\Intersection.h"
#include "..\Scene.h"

#include "..\Utils\rand.h"

#include <vector>
#include <string>

glm::mat4 IdentityMatrix();

float map(float x, float x0, float x1, float y0, float y1);

float Rad2Deg (float AngleFactor);

glm::vec2 GetUniformRandomSample2D(glm::vec2 range_u, glm::vec2 range_v);
void GetRandomSampleDirectionProbability(glm::vec3 orientation, glm::vec3 direction, float& pdf, uint order);
glm::vec3 GetRandomSampleDirectionCosCone(glm::vec3 orientation, float&pdf, uint order);
glm::vec3 GetRandomSampleDirectionCosCone(glm::vec3 orientation, const float u1, const float u2, float&pdf, uint order);
glm::vec3 SampleConeDirection(const glm::vec3& axis, const float& theta, const float& u1, const float& u2, float* pdf);
glm::vec2 ConcentricSampleDisk(float u1, float u2);

float ProbPSA(const SceneSample& from, const SceneSample& to);
float ProbPS(const SceneSample& from, const SceneSample& to);
float ProbA(const SceneSample& from, const SceneSample& to);
float G(const SceneSample& ss1, const SceneSample& ss2);

void GetStratifiedSamples2D(std::vector<glm::vec2>& samples, const glm::vec2& range, const uint numSamples);
glm::vec2 GetUniformSample2D(const glm::vec2& range);

glm::mat3 ComputeTangentSpace(const glm::vec3& n );
glm::vec3 NeverCoLinear(const glm::vec3& v);
glm::vec3 Orthogonal(const glm::vec3 &v);

float map(float x, float x0, float x1, float y0, float y1);

float clamp(const float& x, const float& low, const float& high);

bool IntersectRayBox(const Ray& ray, glm::vec3 boxMin, glm::vec3 boxMax);

bool IntersectWithBB(const CTriangle& triangle, const Ray& ray);

bool IntersectRayTriangle(const Ray& ray, glm::vec3 v0, glm::vec3 v1, glm::vec3 v2, float &t);

std::string AsString(glm::vec3 v);
std::string AsString(glm::vec4 v);

void PlaneHammersley(float *result, int n);

uint GetBiggestSquareNumSmallerThan(uint num);

float G(glm::vec3 p1, glm::vec3 n1, glm::vec3 p2, glm::vec3 n2);
float G_A(glm::vec3 p_avpl, glm::vec3 n_avpl, glm::vec3 p_point, glm::vec3 n_point);

float Luminance(glm::vec3 v);
float Luminance(glm::vec4 v);

#endif