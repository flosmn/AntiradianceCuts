#include "Util.h"

#include <glm/gtx/transform.hpp>
#include <glm/gtx/intersect.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/random.hpp>

#include <sstream>
#include <iostream>

#include <algorithm>
#include <math.h>

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

glm::vec3 hue_colormap(const float v, const float range_min, const float range_max)
{
	const float H = clamp((range_max - v)/(range_max - range_min), 0.f, 1.f) * 4.f;
	const float X = clamp(1.f - abs(std::fmod(H, 2.f) - 1.f), 0.f, 1.f); 

	if(H < 1.f)
		return glm::vec3(1.f, X, 0.f);
	else if(H < 2.f)
		return glm::vec3(X, 1.f, 0.f);
	else if(H < 3.f)
		return glm::vec3(0.f, 1.f, X);
	else
		return glm::vec3(0.f, X, 1.f);
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

glm::vec3 GetRandomSampleDirectionCosCone(glm::vec3 orientation, const float u1, const float u2, float &pdf, uint order)
{
	glm::mat3 TS_to_WS = ComputeTangentSpace(orientation);

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
		cos(theta)));
		
	const float cos_theta = glm::dot(sampleDir, glm::vec3(0.f, 0.f, 1.f));
	pdf = ((float)order + 1) / (2 * PI) * std::powf(cos_theta, (float)order);
			
	glm::vec3 direction = glm::normalize(TS_to_WS * sampleDir);

	if(glm::dot(direction, orientation) < 0.f)
		std::cout << "wrong direction" << std::endl;

	return direction;
}

glm::mat3 ComputeTangentSpace(const glm::vec3& n)
{
	glm::vec3 t = NeverCoLinear(n);
	const glm::vec3 U = glm::normalize( glm::cross( t, n ) );
	const glm::vec3 V = glm::normalize( glm::cross( U, n ) );
	return glm::mat3(U, V, n);
}

void GetRandomSampleDirectionProbability(glm::vec3 orientation, glm::vec3 direction, float& pdf, uint order)
{
	glm::mat3 TS_to_WS = ComputeTangentSpace(orientation);
	glm::mat3 WS_to_TS = glm::transpose(TS_to_WS);
		
	glm::vec3 directionTS = WS_to_TS * direction;
	
	const float cos_theta = glm::dot(directionTS, glm::vec3(0.f, 0.f, 1.f));
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
	return 0.2126f * std::max(v.r, 0.f) + 0.7152f * std::max(v.g, 0.f) + 0.0722f * std::max(v.b, 0.f);
}

float Luminance(glm::vec4 v)
{
	return Luminance(glm::vec3(v));
}

float ProbPSA(const SceneSample& from, const SceneSample& to, const float pdfSA)
{
	glm::vec3 direction = glm::normalize(to.position - from.position);
	const float pdf = pdfSA / glm::dot(from.normal, direction);
	if(pdf <= 0.f)
		std::cout << "ProbPSA: pdf <= 0" << std::endl;
	return pdf;
}

float ProbA(const SceneSample& from, const SceneSample& to, const float pdfSA)
{
	const float pdf = G(from, to) * ProbPSA(from, to, pdfSA);
	if(pdf <= 0.f)
		std::cout << "ProbA: pdf <= 0" << std::endl;
	return pdf;
}

float G(const SceneSample& ss1, const SceneSample& ss2)
{
	const glm::vec3 direction12 = glm::normalize(ss2.position - ss1.position);
	const float cos_theta1 = glm::dot(ss1.normal, direction12);
	const float cos_theta2 = glm::dot(ss2.normal, -direction12);
	const float dist = glm::length(ss2.position - ss1.position);

	return cos_theta1 * cos_theta2 / (dist * dist);
}

glm::vec4 Phong(const glm::vec3& from, const glm::vec3& over, const glm::vec3& to, const glm::vec3& n, MATERIAL* mat)
{
	const glm::vec3 w_i = glm::normalize(from - over);
	const glm::vec3 w_o = glm::normalize(to - over);
	return Phong(w_i, w_o, n, mat);
}

glm::vec4 Phong(const glm::vec3& w_i, const glm::vec3& w_o, const glm::vec3& n, MATERIAL* mat)
{
	glm::vec4 diffuse = ONE_OVER_PI * mat->diffuse;
	glm::vec3 refl = reflect(w_i, n);
	const float cos = std::max(0.f, glm::dot(refl, w_o));
	glm::vec4 specular = 0.5f * ONE_OVER_PI * (mat->exponent+2) * powf(cos, mat->exponent) * mat->specular;
	const glm::vec4 res = diffuse + specular;

	return res;
}

glm::vec3 reflect(const glm::vec3& v, const glm::vec3& n)
{
	const float cos_theta = glm::dot(v,n);
	if(cos_theta < 0.f)
		std::cout << "reflect wrong" << std::endl;

	return glm::normalize(2 * cos_theta * n - v);
}

glm::vec3 SamplePhong(const glm::vec3& w_o, const glm::vec3& n, MATERIAL* mat, float& pdf, bool MIS)
{
	if(!MIS)
	{
		glm::vec3 direction = GetRandomSampleDirectionCosCone(n, Rand01(), Rand01(), pdf, 1);
		if(pdf <= 0.f)
			std::cout << "SamplePhong: pdf <= 0.f" << std::endl;
		return direction;
	}
	else
	{
		const float k_d = 1.f/3.f * (mat->diffuse.x + mat->diffuse.y + mat->diffuse.z);
		const float k_s = 1.f/3.f * (mat->specular.x + mat->specular.y + mat->specular.z);

		const float p_d = k_d / (k_d + k_s);
			
		if(Rand01() <= p_d)
		{
			// sample diffuse
			float p = 0.f;
			glm::vec3 direction = GetRandomSampleDirectionCosCone(n, Rand01(), Rand01(), p, 1);
			const float p_1 = p_d * p;
			
			glm::vec3 w_i = reflect(w_o, n);
			GetRandomSampleDirectionProbability(w_i, direction, p, (uint)mat->exponent);
			const float p_2 = (1.f - p_d) * p;
			
			const float w = p_1 / (p_1 + p_2);

			pdf = p_1 / w;
			
			if(pdf <= 0.f)
				std::cout << "SamplePhong: pdf <= 0.f" << std::endl;
			return direction;
		}
		else
		{
			// sample specular
			float p = 0.f;
			glm::vec3 w_i = reflect(w_o, n);
			glm::vec3 direction = GetRandomSampleDirectionCosCone(w_i, Rand01(), Rand01(), p, (uint)mat->exponent);
			if(glm::dot(direction, n) <= 0.f)
			{
				pdf = 0.f;
				return n;
			}
			
			const float p_2 = (1.f - p_d) * p;
			
			GetRandomSampleDirectionProbability(n, direction, p, 1);
			const float p_1 = p_d * p;
			
			const float w = p_2 / (p_1 + p_2);

			pdf = p_2 / w;

			if(pdf <= 0.f)
				std::cout << "SamplePhong: pdf <= 0.f" << std::endl;
			return direction;
		}
	}
}

float PhongPdf(const glm::vec3& w_i, const glm::vec3& w_o, const glm::vec3& n, MATERIAL* mat, bool MIS)
{
	if(!MIS)
	{
		float p = 0.f;
		GetRandomSampleDirectionProbability(n, w_o, p, 1);
		if(p <= 0.f)
				std::cout << "PhongPdf: pdf <= 0.f" << std::endl;
		return p;
	}
	else
	{
		const float k_d = 1.f/3.f * (mat->diffuse.x + mat->diffuse.y + mat->diffuse.z);
		const float k_s = 1.f/3.f * (mat->specular.x + mat->specular.y + mat->specular.z);

		const float p_d = k_d / (k_d + k_s);

		// sample diffuse
		float p = 0.f;
		GetRandomSampleDirectionProbability(n, w_o, p, 1);
		const float p_1 = p_d * p;

		glm::vec3 refl = reflect(w_i, n);
		GetRandomSampleDirectionProbability(refl, w_o, p, (uint)mat->exponent);
		const float p_2 = (1.f-p_d) * p;

		const float pdf = p_1 + p_2;
		if(pdf <= 0.f)
			std::cout << "PhongPdf: pdf <= 0.f" << std::endl;
		return pdf;
	}
}

float PhongPdf(const glm::vec3& from, const glm::vec3& over, const glm::vec3& to, const glm::vec3& n, MATERIAL* mat, bool MIS)
{
	const glm::vec3 w_i = glm::normalize(from - over);
	const glm::vec3 w_o = glm::normalize(to - over);
	return PhongPdf(w_i, w_o, n, mat, MIS);
}
