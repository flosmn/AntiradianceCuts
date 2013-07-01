#ifndef UTIL_H
#define UTIL_H

typedef unsigned int uint;

#include "glm/glm.hpp"

#include "..\Defines.h"
#include "..\Ray.h"
#include "..\Intersection.h"
#include "..\SceneSample.h"
#include "..\Scene.h"

#include "..\Utils\rand.h"
#include <limits>

#include <vector>
#include <string>

glm::mat4 IdentityMatrix();

float map(float x, float x0, float x1, float y0, float y1);

float Rad2Deg (float AngleFactor);
glm::vec3 hue_colormap(const float v, const float range_min, const float range_max);

glm::vec2 GetUniformRandomSample2D(glm::vec2 range_u, glm::vec2 range_v);
void GetRandomSampleDirectionProbability(glm::vec3 orientation, glm::vec3 direction, float& pdf, uint order);

glm::vec3 GetRandomSampleDirectionCosCone(glm::vec3 orientation, const float u1, const float u2, float&pdf, uint order);
void GetStratifiedDirections(std::vector<glm::vec3>& directions, std::vector<float>& pdfs, int numDirections, glm::vec3 orientation, uint order);

glm::vec3 SampleConeDirection(const glm::vec3& axis, const float& theta, const float& u1, const float& u2, float* pdf);

glm::vec2 ConcentricSampleDisk(float u1, float u2);

glm::mat3 ComputeTangentSpace(const glm::vec3& n);

float ProbPSA(const SceneSample& from, const SceneSample& to, const float pdfSA);
float ProbA(const SceneSample& from, const SceneSample& to, const float pdfSA);
float G(const SceneSample& ss1, const SceneSample& ss2);

glm::vec3 reflect(const glm::vec3& v, const glm::vec3& norm);

glm::vec4 Phong(const glm::vec3& from, const glm::vec3& over, const glm::vec3& to, const glm::vec3& n, MATERIAL* mat);
glm::vec4 Phong(const glm::vec3& w_i, const glm::vec3& w_o, const glm::vec3& n, MATERIAL* mat);

float PhongPdf(const glm::vec3& w_i, const glm::vec3& w_o, const glm::vec3& n, MATERIAL* mat, bool MIS);
float PhongPdf(const glm::vec3& from, const glm::vec3& over, const glm::vec3& to, const glm::vec3& n, MATERIAL* mat, bool MIS);

glm::vec3 SamplePhong(const glm::vec3& w_o, const glm::vec3& n, MATERIAL* mat, float& pdf, bool MIS);

void GetStratifiedSamples2D(std::vector<glm::vec2>& samples, const glm::vec2& range, const uint numSamples);
glm::vec2 GetUniformSample2D(const glm::vec2& range);

glm::mat3 ComputeTangentSpace(const glm::vec3& n );
glm::vec3 NeverCoLinear(const glm::vec3& v);
glm::vec3 Orthogonal(const glm::vec3 &v);

float map(float x, float x0, float x1, float y0, float y1);

float clamp(const float& x, const float& low, const float& high);

bool IntersectRayBox(const Ray& ray, glm::vec3 boxMin, glm::vec3 boxMax);

bool IntersectWithBB(const Triangle& triangle, const Ray& ray);

bool IntersectRayTriangle(const Ray& ray, glm::vec3 v0, glm::vec3 v1, glm::vec3 v2, float &t);

std::string AsString(glm::vec3 v);
std::string AsString(glm::vec4 v);

void PlaneHammersley(float *result, int n);

uint GetBiggestSquareNumSmallerThan(uint num);

float G(glm::vec3 p1, glm::vec3 n1, glm::vec3 p2, glm::vec3 n2);
float G_A(glm::vec3 p_avpl, glm::vec3 n_avpl, glm::vec3 p_point, glm::vec3 n_point);

//float Luminance(glm::vec3 v);
//float Luminance(glm::vec4 v);

float Average(glm::vec3 v);
float Average(glm::vec4 v);

glm::vec3 GetIrradiance(const AVPL& avpl, const SceneSample& ss);
glm::vec3 GetAntiirradiance(const AVPL& avpl, const SceneSample& ss, float N);

template<typename T>
inline bool is_nan(T value)
{
	return value != value;
}

template<typename T>
inline bool is_inf(T value)
{
	return std::numeric_limits<T>::has_infinity && value == std::numeric_limits<T>::infinity();
}

#endif