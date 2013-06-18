#include "CPathTracingIntegrator.h"

#include <omp.h>

#include "Scene.h"
#include "CCamera.h"
#include "CImagePlane.h"
#include "Utils\Util.h"

#include <iostream>

CPathTracingIntegrator::CPathTracingIntegrator(Scene* scene, CImagePlane* imagePlane)
	: m_scene(scene), m_imagePlane(imagePlane)
{
}

CPathTracingIntegrator::~CPathTracingIntegrator()
{
}

void CPathTracingIntegrator::Integrate(uint numPaths, bool MIS)
{	
	//omp_set_num_threads(8);

	#pragma omp parallel for
	for(int k = 0; k < (int)numPaths; ++k)
	{		
		glm::vec2 pixel;
		Ray r(glm::vec3(0.f), glm::vec3(0.f));

		#pragma omp critical
		{
			r = m_scene->GetCamera()->GetEyeRay(pixel);
		}

		SceneSample y[100];
		float p_A[100];

		glm::vec4 f_x = glm::vec4(0.f);
		glm::vec4 color = glm::vec4(0.f);

		float t = 0;
		Intersection intersection;
		if(m_scene->IntersectRayScene(r, &t, &intersection, CTriangle::FRONT_FACE))
		{
			SceneSample ss(intersection);

			y[0] = m_scene->GetCamera()->GetCameraAsSceneSample();
			p_A[0] = 1.f;
			y[1] = ss;
			p_A[1] = 1.f; //m_scene->GetCamera()->GetProbWrtArea(ss);

			f_x = glm::vec4(1.f); //glm::vec4(G(y[0], y[1]) / p_A[1]);
		}
		else
		{
			continue;
		}

		bool terminate = false;
		int i = 0;
		while(!terminate)
		{
			terminate = true;
			i++;

			// check implicit path
			if(glm::length(m_scene->GetMaterial(intersection)->emissive) > 0.f)
			{
				float w = 0.f;
				if(i > 1)
				{
					const float pdfSA = PhongPdf(y[i-2].position, y[i-1].position, y[i].position, y[i-1].normal, 
						m_scene->GetMaterial(y[i-1]), MIS);
					
					if(pdfSA <= 0.f)
					{
						terminate = true;
						continue;
					}
					
					const float p = ProbA(y[i-1], y[i], pdfSA);
					w = p / (p_A[i] + p);
				}
				else
				{
					w = 1.f;
				}
				
				color += w * f_x * m_scene->GetMaterial(intersection)->emissive;
			}

			// make explicit connection
			SceneSample ls_sample;
			m_scene->SampleLightSource(ls_sample);
			if(m_scene->Visible(y[i], ls_sample))
			{
				const float pdfSA = PhongPdf(y[i-1].position, y[i].position, ls_sample.position, y[i].normal, 
					m_scene->GetMaterial(y[i]), MIS);
				
				if(pdfSA <= 0.f)
				{
					terminate = true;
					continue;
				}
				
				const float p_A = ProbA(y[i], ls_sample, pdfSA);
				const float w = ls_sample.pdf / (ls_sample.pdf + p_A);
				const glm::vec4 L_e = m_scene->GetMaterial(ls_sample)->emissive;
			
				const glm::vec4 BRDF = Phong(y[i-1].position, y[i].position, ls_sample.position, y[i].normal, 
					m_scene->GetMaterial(y[i]));
							
				color += w * f_x * BRDF * G(ls_sample, y[i]) * L_e/ls_sample.pdf;
			}

			// continue path
			float t_prob = 0.5f;
			if(Rand01() > t_prob)
			{
				terminate = true;
			}
			else
			{
				float pdf = 0;
				glm::vec3 direction;
				
				#pragma omp critical
				{
					direction = SamplePhong(glm::normalize(y[i-1].position - y[i].position), y[i].normal, 
						m_scene->GetMaterial(y[i]), pdf, MIS);
				}
				
				if(glm::dot(direction, y[i].normal) <= 0.f)
				{
					std::cout << "cos_theta <= 0.f" << std::endl;
					terminate = true;
					continue;
				}
				
				Ray ray(y[i].position, direction);

				if(m_scene->IntersectRayScene(ray, &t, &intersection, CTriangle::FRONT_FACE) && pdf > 0.f)
				{
					SceneSample ss(intersection);
					y[i+1] = ss;
					
					const float temp = ProbA(y[i], y[i+1], pdf);
					if(temp <= 0.f)
					{
						terminate = true;
						continue;
					}

					p_A[i+1] = p_A[i] + temp;
					
					const glm::vec4 BRDF = Phong(y[i-1].position, y[i].position, y[i+1].position, y[i].normal, 
						m_scene->GetMaterial(y[i]));

					f_x = 1.f/(ProbPSA(y[i], y[i+1], pdf) * t_prob) * f_x * BRDF;
				}
				else
				{
					terminate = true;
				}
			}
		}

		#pragma omp critical
		{
			m_imagePlane->AddSample(pixel, color);
		}
	}
}
