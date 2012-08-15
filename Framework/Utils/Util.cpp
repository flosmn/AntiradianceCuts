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

float Rad2Deg (float AngleFactor) {
	static float ratio = 180.0f / PI;
	return AngleFactor * ratio;
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

void GetRandomSampleDirectionProbability(glm::vec3 orientation, glm::vec3 direction, float& pdf, uint order)
{
	glm::vec3 up = glm::vec3(0.0f, 1.0f, 0.0f);
	if(glm::abs(glm::dot(up, orientation)) > 0.009) up = glm::vec3(0.0f, 0.0f, 1.0f); 
	
	glm::mat4 transformDir = glm::lookAt(glm::vec3(0.f), orientation, up);
	
	glm::vec3 orientationLS = glm::vec3(transformDir * glm::vec4(orientation, 0.f));
	glm::vec3 directionLS = glm::vec3(transformDir * glm::vec4(direction, 0.f));
	
	const float cos_theta = glm::dot(directionLS, glm::vec3(0.f, 0.f, -1.f));
	pdf = ((float)order + 1) / (2 * PI) * std::powf(cos_theta, (float)order);
}

glm::vec2 ConcentricSampleDisk(float u1, float u2) {
    float r, theta;
    // Map uniform random numbers to $[-1,1]^2$
    float sx = 2 * u1 - 1;
    float sy = 2 * u2 - 1;

    // Map square to $(r,\theta)$

    // Handle degeneracy at the origin
    if (sx == 0.0 && sy == 0.0) {
        return glm::vec2(0.f, 0.f);
    }
    if (sx >= -sy) {
        if (sx > sy) {
            // Handle first region of disk
            r = sx;
            if (sy > 0.0) theta = sy/r;
            else          theta = 8.0f + sy/r;
        }
        else {
            // Handle second region of disk
            r = sy;
            theta = 2.0f - sx/r;
        }
    }
    else {
        if (sx <= sy) {
            // Handle third region of disk
            r = -sx;
            theta = 4.0f - sy/r;
        }
        else {
            // Handle fourth region of disk
            r = -sy;
            theta = 6.0f + sx/r;
        }
    }
    theta *= PI / 4.f;

	return glm::vec2(r * cosf(theta), r * sinf(theta));
}

glm::vec3 SampleConeDirection(const glm::vec3& axis, const float& theta, const float& u1, const float& u2, float* pdf)
{
	glm::vec3 direction = glm::vec3(0.f);

	// create a uniform random sample on the disk with radius tan(theta)
	// where theta = PI / N
	const float r = tanf(theta);
	glm::vec2 diskSample = ConcentricSampleDisk(u1, u2);
	diskSample = r * diskSample;

	*pdf = 1.f/(PI * r * r); // uniform pdf

	// cone origin (0,0,0), diskSample (x,z,1)
	glm::vec3 dirLocal = glm::normalize(glm::vec3(diskSample.x, diskSample.y, 1));

	// transform s.t. the direction is in the same space
	// as the antiradiance direction
	glm::mat3 M = ComputeTangentSpace(axis);

	direction = glm::transpose(M) * dirLocal;
	
	return glm::normalize(direction);
}

void GetStratifiedSamples2D(std::vector<glm::vec2>& samples, const glm::vec2& range, const uint numSamples)
{
	uint N = GetBiggestSquareNumSmallerThan(numSamples);
	uint n = uint(sqrt(float(N)));
	uint remaining = numSamples - N;

	// create stratified samples
	const float grid_size_x = range.x / float(n);
	const float grid_size_y = range.y / float(n);
	for(uint i = 0; i < n; ++i)
	{
		for(uint j = 0; j < n; ++j)
		{
			glm::vec2 sample = glm::vec2(
				(float(i)+Rand01()) * grid_size_x,
				(float(j)+Rand01()) * grid_size_y);
			samples.push_back(sample);
		}
	}

	// create remaining uniformly distributed random samples
	for(uint i = 0; i < remaining; ++i)
	{
		glm::vec2 sample = glm::vec2(Rand01() * range.x, Rand01() * range.y);
		samples.push_back(sample);
	}
}

glm::vec2 GetUniformSample2D(const glm::vec2& range)
{
	return glm::vec2(Rand01() * range.x, Rand01() * range.y);
}

glm::mat3 ComputeTangentSpace(const glm::vec3& n ) {
	glm::vec3 nev = NeverCoLinear(n);

	glm::vec3 T = glm::normalize( glm::cross( nev, n ) );
	glm::vec3 B = glm::cross( n, T );
	
	return glm::transpose(glm::mat3( T, B, n ));
}

glm::vec3 NeverCoLinear(const glm::vec3& v)
{
	glm::vec3 n;
	n.x = v.y;
	n.y = v.z;
	n.z = -v.x;

	return n;
}

glm::vec3 Orthogonal(const glm::vec3& v)
{
	glm::vec3 n = NeverCoLinear(v);
	return glm::normalize(glm::cross(n, v));
}

float map(float x, float x0, float x1, float y0, float y1) {
	return y1 - (x1 - x) * (y1 - y0) / (x1 - x0);
}

bool IntersectWithBB(const CTriangle& triangle, const Ray& ray)
{
	glm::vec3 p1 = triangle.P0();
	glm::vec3 p2 = triangle.P1();
	glm::vec3 p3 = triangle.P2();
	
	float minX = glm::min(p1.x, glm::min(p2.x, p3.x));
	float minY = glm::min(p1.y, glm::min(p2.y, p3.y));
	float minZ = glm::min(p1.z, glm::min(p2.z, p3.z));

	float maxX = glm::max(p1.x, glm::max(p2.x, p3.x));
	float maxY = glm::max(p1.y, glm::max(p2.y, p3.y));
	float maxZ = glm::max(p1.z, glm::max(p2.z, p3.z));

	glm::vec3 boxMin = glm::vec3(minX, minY, minZ);
	glm::vec3 boxMax = glm::vec3(maxX, maxY, maxZ);

	return IntersectRayBox(ray, boxMin, boxMax);
}

bool IntersectRayBox(const Ray& ray, glm::vec3 boxMin, glm::vec3 boxMax) 
{
	glm::vec3 rayDirInv = 1.0f / ray.d;

	glm::vec3 slabMin = (boxMin - ray.o) * rayDirInv;
	glm::vec3 slabMax = (boxMax - ray.o) * rayDirInv;

	glm::vec3 absMin = glm::min(slabMin, slabMax);
	glm::vec3 absMax = glm::max(slabMin, slabMax);
		
	float t0 = glm::max(0.0f, glm::max(absMin.x, glm::max(absMin.y, absMin.z)));
	float t1 = glm::min(absMax.x, glm::min(absMax.y, absMax.z));

	bool intersection = t0 <= t1;
	return intersection;
}

bool IntersectRayTriangle(const Ray& ray, glm::vec3 v0, glm::vec3 v1, glm::vec3 v2, float &t)
{
	float epsilon = 0.0001f;

	glm::vec3 vec_p, vec_t, vec_q;
	glm::vec3 e1 = v2 - v0;
	glm::vec3 e2 = v1 - v0;

	vec_p = glm::cross(ray.d, e2);

	float det = glm::dot(e1, vec_p);

	if(det < epsilon && det > -epsilon) return false;
	
	vec_t = ray.o - v0;

	float u = glm::dot(vec_t, vec_p);

	if(u < 0.0f || u > det) return false;

	vec_q = glm::cross(vec_t, e1);

	float v = glm::dot(ray.d, vec_q);

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

float clamp(const float& x, const float& low, const float& high)
{
	return std::min(std::max(low, x), high);
}

uint GetBiggestSquareNumSmallerThan(uint num)
{
	uint t = 1;
	while(t*t <= num) ++t;
	--t;
	return t*t;
}

float G(glm::vec3 p1, glm::vec3 n1, glm::vec3 p2, glm::vec3 n2)
{
	glm::vec3 n_1 = glm::normalize(n1);
	glm::vec3 n_2 = glm::normalize(n2);
	glm::vec3 w = glm::normalize(p2 - p1);

	const float cos_theta_1 = clamp(glm::dot(n_1, w), 0, 1);
	const float cos_theta_2 = clamp(glm::dot(n_2, -w), 0, 1);

	const float dist = glm::length(p2 - p1);
	
	return (cos_theta_1 * cos_theta_2) / (dist * dist);
}

float G_A(glm::vec3 p_avpl, glm::vec3 n_avpl, glm::vec3 p_point, glm::vec3 n_point)
{
	const glm::vec3 w = glm::normalize(p_avpl - p_point);
	const float cos_theta = clamp(glm::dot(n_point, w), 0, 1);
	const float dist = glm::length(p_avpl - p_point);
	
	return cos_theta / (dist * dist);
}

float Luminance(glm::vec3 v)
{
	return 0.2126f * v.r + 0.7152f * v.g + 0.0722f * v.b;
}

float Luminance(glm::vec4 v)
{
	return Luminance(glm::vec3(v));
}