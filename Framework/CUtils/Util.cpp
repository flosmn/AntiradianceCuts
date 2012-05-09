#include "Util.h"

#include <glm/gtx/transform.hpp>
#include <glm/gtx/intersect.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/random.hpp>

#include <sstream>
#include <iostream>

glm::mat4 IdentityMatrix() 
{
	return glm::mat4(
		1.0f, 0.0f, 0.0f, 0.0f,
		0.0f, 1.0f, 0.0f, 0.0f,
		0.0f, 0.0f, 1.0f, 0.0f,
		0.0f, 0.0f, 0.0f, 1.0f );
}

float Rad2Deg (float Angle) {
	static float ratio = 180.0f / PI;
	return Angle * ratio;
}

glm::vec3 GetArbitraryUpVector(glm::vec3 v)
{
	// dot(temp, v) != 1
	glm::vec3 temp = glm::vec3(1.0f/(v.x+1), -v.z, v.y);

	// dot(temp2, v) = 0
	glm::vec3 temp2 = glm::cross(temp, v);

	return glm::cross(temp2, v);
}

glm::vec2 GetUniformRandomSample2D(glm::vec2 range_u, glm::vec2 range_v) {
	float u = Rand01();
	float v = Rand01();
	u = map(u, 0, 1, range_u.x, range_u.y);
	v = map(v, 0, 1, range_v.x, range_v.y);
	return glm::vec2(u, v);
}

glm::vec3 GetRandomSampleDirectionCosCone(glm::vec3 orientation, float&pdf, uint order)
{
	glm::vec3 direction = glm::vec3(0.f);

	do {
		float xi_1 = glm::linearRand(0.f, 1.f);
		float xi_2 = glm::linearRand(0.f, 1.f);
		
		direction = GetRandomSampleDirectionCosCone(orientation, xi_1, xi_2, pdf, order);

	} while (pdf < 0.0001f);
	
	return direction;
}

glm::vec3 GetRandomSampleDirectionCosCone(glm::vec3 orientation, const float u1, const float u2, float&pdf, uint order)
{
	glm::vec3 sampleDir = glm::vec3(0.f);

	float xi_1 = u1;
	float xi_2 = u2;
		
	// cos-sampled point with orientation (0, 0, 1)
	float power = 1.0f / ((float)order + 1.0f);
	float theta = acos(pow(xi_1, power));
	float phi = 2 * PI * xi_2;

	sampleDir = glm::normalize(glm::vec3(
		sin(theta) * cos(phi),
		sin(theta) * sin(phi),
		-cos(theta)) );
		
	const float cos_theta = glm::dot(sampleDir, glm::vec3(0.f, 0.f, -1.f));
	pdf = ((float)order + 1) / (2 * PI) * std::powf(cos_theta, (float)order);
			
	glm::vec4 directionTemp = glm::vec4(sampleDir, 0.0f);
	
	glm::vec3 up = glm::vec3(0.0f, 1.0f, 0.0f);
		
	// make sure that orientation and up are not similar
	if(glm::abs(glm::dot(up, orientation)) > 0.009) up = glm::vec3(0.0f, 0.0f, 1.0f); 

	// transformation to get the sampled oriented after 
	// the orientation vector
	glm::mat4 transformDir = glm::inverse(glm::lookAt(glm::vec3(0.f), orientation, up));
	
	directionTemp = transformDir * directionTemp;
	
	glm::vec3 direction = glm::normalize(glm::vec3(directionTemp));

	return direction;
}

float map(float x, float x0, float x1, float y0, float y1) {
	return y1 - (x1 - x) * (y1 - y0) / (x1 - x0);
}

bool IntersectWithBB(Triangle triangle, Ray ray)
{
	glm::vec3 p1 = triangle.GetPoints()[0];
	glm::vec3 p2 = triangle.GetPoints()[1];
	glm::vec3 p3 = triangle.GetPoints()[2];
	
	float minX = glm::min(p1.x, glm::min(p2.x, p3.x));
	float minY = glm::min(p1.y, glm::min(p2.y, p3.y));
	float minZ = glm::min(p1.z, glm::min(p2.z, p3.z));

	float maxX = glm::max(p1.x, glm::max(p2.x, p3.x));
	float maxY = glm::max(p1.y, glm::max(p2.y, p3.y));
	float maxZ = glm::max(p1.z, glm::max(p2.z, p3.z));

	glm::vec3 boxMin = glm::vec3(minX, minY, minZ);
	glm::vec3 boxMax = glm::vec3(maxX, maxY, maxZ);

	return IntersectRayBox(ray.GetOrigin(), ray.GetDirection(), boxMin, boxMax);
}

bool IntersectRayBox(glm::vec3 rayPos, glm::vec3 rayDir, glm::vec3 boxMin, glm::vec3 boxMax) 
{
	glm::vec3 rayDirInv = 1.0f / rayDir;

	glm::vec3 slabMin = (boxMin - rayPos) * rayDirInv;
	glm::vec3 slabMax = (boxMax - rayPos) * rayDirInv;

	glm::vec3 absMin = glm::min(slabMin, slabMax);
	glm::vec3 absMax = glm::max(slabMin, slabMax);
		
	float t0 = glm::max(0.0f, glm::max(absMin.x, glm::max(absMin.y, absMin.z)));
	float t1 = glm::min(absMax.x, glm::min(absMax.y, absMax.z));

	bool intersection = t0 <= t1;
	return intersection;
}

bool IntersectRayTriangle(glm::vec3 origin, glm::vec3 direction, glm::vec3 v0, glm::vec3 v1, glm::vec3 v2, float &t)
{
	float epsilon = 0.0001f;

	glm::vec3 vec_p, vec_t, vec_q;
	glm::vec3 e1 = v1 - v0;
	glm::vec3 e2 = v2 - v0;

	vec_p = glm::cross(direction, e2);

	float det = glm::dot(e1, vec_p);

	if(det < epsilon && det > -epsilon) return false;
	
	vec_t = origin - v0;

	float u = glm::dot(vec_t, vec_p);

	if(u < 0.0f || u > det) return false;

	vec_q = glm::cross(vec_t, e1);

	float v = glm::dot(direction, vec_q);

	if(v < 0.0f || u + v > det) return false;

	t = glm::dot(e2, vec_q);
	t *= 1.0f/det;

	return true;
}

std::string AsString(glm::vec3 v)
{
	std::ostringstream temp;
	temp << "(" << v.x << ", " << v.y << ", " << v.z << ")";
	std::string str(temp.str());
	return str;
}

std::string AsString(glm::vec4 v)
{
	std::ostringstream temp;
	temp << "(" << v.x << ", " << v.y << ", " << v.z << ", " << v.w << ")";
	std::string str(temp.str());
	return str;
}

void PlaneHammersley(float *result, int n)
{
	float p, u, v;
	int k, kk, pos;
	for (k=0, pos=0 ; k<n ; k++)
	{
		u = 0;
		for (p=0.5f, kk=k ; kk ; p*=0.5f, kk>>=1)
			if (kk & 1) // kk mod 2 == 1
				u += p;
		v = (k + 0.5f) / n;
		result[pos++] = u;
		result[pos++] = v;
	}
}