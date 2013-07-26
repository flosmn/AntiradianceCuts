#ifndef SAMPLER_H_
#define SAMPLER_H_

#include "Util.h"
#include "Defines.h"

#include <random>
#include <memory>
#include <chrono>

#include <omp.h>

static inline glm::vec3 sampleCosCone(glm::vec3 const& normal, glm::vec2 const& sample, float &pdf, float order)
{
	const glm::mat3 TS_to_WS = tangentSpace(normal);
	
	// cos-sampled point with orientation (0, 0, 1)
	const float power = 1.0f / (order + 1.0f);
	const float theta = acos(pow(sample.x, power));
	const float phi = TWO_PI * sample.y;

	const glm::vec3 sampleDir = glm::normalize(glm::vec3(
		sin(theta) * cos(phi),
		sin(theta) * sin(phi),
		cos(theta)));
		
	const float cos_theta = sampleDir.z;
	pdf = (order + 1.f) * ONE_OVER_TWO_PI * std::powf(cos_theta, order);
			
	const glm::vec3 direction = glm::normalize(TS_to_WS * sampleDir);
#ifdef _DEBUG
	if(glm::dot(direction, normal) < 0.f)
		std::cout << "wrong direction" << std::endl;
#endif
	return direction;
}

class Sampler
{
public:
	Sampler(int seed) : 
		m_engine(std::mt19937_64(seed)), 
		m_dist(std::uniform_real_distribution<float>(0.f, 1.f))
	{ }

	Sampler() :
		m_engine(std::mt19937_64((omp_get_thread_num() + 1) * static_cast<uint64_t>(std::chrono::system_clock::to_time_t(std::chrono::system_clock::now())))), 
		m_dist(std::uniform_real_distribution<float>(0.f, 1.f))
	{ }

	// returns a uniformly distributed random number in [0, 1]
	inline float getSample() {
		return m_dist(m_engine);
	}
	
private:
	std::mt19937_64 m_engine;
	std::uniform_real_distribution<float> m_dist;
};

#endif // SAMPLER_H_
