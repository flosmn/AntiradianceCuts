#ifndef AVPL_SHOOTER_H_
#define AVPL_SHOOTER_H_

#include "Avpl.h"
#include "Scene.h"
#include "AreaLight.h"
#include "CConfigManager.h"
#include "Sampler.h"
#include "Timer.h"
#include "Brdf.h"

#include <omp.h>

#include <vector>
#include <memory>
#include <iostream>

class AvplShooter
{
public:
	AvplShooter(Scene* scene, CConfigManager* confManager) :
		m_scene(scene), m_confManager(confManager) 
	{
		m_num_threads = omp_get_max_threads();
		omp_set_num_threads(m_num_threads);
		Sampler sampler;
		for (int i = 0; i < m_num_threads; ++i) {
			m_sampler.push_back(Sampler(int(i * sampler.getSample() * 1000)));
		}
	}

	void shoot(std::vector<Avpl>& avpls_direct, std::vector<Avpl>& avpls_indirect, int numPaths) {
#ifdef _DEBUG
		Sampler sampler;
		for (int i = 0; i < numPaths; ++i) {
			createPath(avpls_direct, avpls_indirect, m_sampler[0]);
		}
#else
		std::vector<std::vector<Avpl>> avpls_direct_per_thread(m_num_threads);
		std::vector<std::vector<Avpl>> avpls_indirect_per_thread(m_num_threads);
		
		for (int i = 0; i < m_num_threads; ++i) {
			avpls_direct_per_thread[i].clear();
			avpls_indirect_per_thread[i].clear();
		}
		
		#pragma omp parallel for
		for (int i = 0; i < numPaths; ++i) {
			createPath(avpls_direct_per_thread[omp_get_thread_num()], 
				avpls_indirect_per_thread[omp_get_thread_num()], 
				m_sampler[omp_get_thread_num()]);
		}
		#pragma omp barrier
		
		int num_avpls_direct = 0;
		int num_avpls_indirect = 0;
		for (int i = 0; i < m_num_threads; ++i) {
			num_avpls_direct += avpls_direct_per_thread[i].size();
			num_avpls_indirect += avpls_indirect_per_thread[i].size();
		}

		avpls_direct.resize(num_avpls_direct);
		avpls_indirect.resize(num_avpls_indirect);
		
		int offset_direct = 0;
		int offset_indirect = 0;
		for (int i = 0; i < m_num_threads; ++i) {
			std::copy(avpls_direct_per_thread[i].begin(), avpls_direct_per_thread[i].end(), avpls_direct.begin() + offset_direct);
			std::copy(avpls_indirect_per_thread[i].begin(), avpls_indirect_per_thread[i].end(), avpls_indirect.begin() + offset_indirect);
			offset_direct += avpls_direct_per_thread[i].size();
			offset_indirect += avpls_indirect_per_thread[i].size();
		}
#endif
	}

private:
	void createPath(std::vector<Avpl>& avpls_direct, std::vector<Avpl>& avpls_indirect, Sampler& sampler) {
		const float rrProb = 0.8f;

		// sample light source
		float pdf = 0.f;
		const glm::vec3 pos = m_scene->getAreaLight()->samplePos(pdf);
		const glm::vec3 normal = m_scene->getAreaLight()->getFrontDirection();
		const glm::vec3 rad = m_scene->getAreaLight()->getRadiance();

		Avpl avpl(pos, normal, rad/pdf, glm::vec3(0.f), glm::vec3(0.f), 0, m_scene->getAreaLight()->getMaterialIndex());
		avpls_direct.push_back(avpl);

		const glm::vec2 sample(sampler.getSample(), sampler.getSample());
		glm::vec3 direction = sampleCosCone(normal, sample, pdf, 1.f);

		while(true) {
			// try to continue the path
			Intersection isect;
			float t;
			const Ray ray(avpl.getPosition() + EPS * avpl.getNormal(), direction);
			if (!m_scene->IntersectRayScene(ray, &t, &isect, Triangle::FRONT_FACE)) {
				break;
			}

			// create new avpl
			const float cos_theta = glm::dot(avpl.getNormal(), direction);
			glm::vec3 contrib = avpl.getIncidentRadiance() * cos_theta / pdf;

			if (avpl.getBounce() > 0) { // if avpl not on light source
				contrib *= phong_eval(-avpl.getIncidentDirection(), direction, 
					avpl.getNormal(), m_scene->GetMaterialBuffer()->GetMaterial(avpl.getMaterialIndex()));
			}

			const glm::vec3 hit_normal = isect.getTriangle()->getNormal();
			avpl = Avpl(isect.getPosition() - EPS * hit_normal, hit_normal, contrib,
					contrib, direction, avpl.getBounce() + 1, isect.getTriangle()->getMaterialIndex()); 

			if (avpl.getBounce() == 1 && m_confManager->GetConfVars()->explicitDirectIllum) {
				avpl.setAntiradiance(glm::vec3(0.f));
			}

			// russian roulette
			if (sampler.getSample() > rrProb) {
				avpl.setIncidentRadiance(glm::vec3(0.f));	
				avpls_indirect.push_back(avpl);
				break;
			}

			// account for russian roulette
			avpl.setIncidentRadiance(1.f / rrProb * avpl.getIncidentRadiance());
			avpls_indirect.push_back(avpl);

			// sample new direction
			const glm::vec2 sample(sampler.getSample(), sampler.getSample());
			direction = phong_sample(-direction, avpl.getNormal(), 
				m_scene->GetMaterialBuffer()->GetMaterial(avpl.getMaterialIndex()), sample, pdf);
			
			if (glm::length(direction) <= 0.f) {
				break;
			}
			if (pdf < 1e-4f) {
				std::cout << "pdf is small: " << pdf << std::endl;
				break;
			}
		}
	}

	Scene* m_scene;
	CConfigManager* m_confManager;

	std::vector<Sampler> m_sampler;
	int m_num_threads;
};

#endif // AVPL_SHOOTER_H_
