#include "CPathTracingIntegrator.h"

#include "Scene.h"
#include "CCamera.h"
#include "CImagePlane.h"
#include "Utils\Util.h"

CPathTracingIntegrator::CPathTracingIntegrator()
{
}

CPathTracingIntegrator::~CPathTracingIntegrator()
{
}

bool CPathTracingIntegrator::Init(Scene* pScene, CImagePlane* pImagePlane)
{
	m_pScene = pScene;
	m_pImagePlane = pImagePlane;

	return true;
}

void CPathTracingIntegrator::Integrate(uint numPaths)
{	
	for(uint k = 0; k < numPaths; ++k)
	{		
		glm::vec2 pixel;
		Ray r = m_pScene->GetCamera()->GetEyeRay(pixel);

		SceneSample y[100];
		float p_A[100];

		glm::vec4 f_x = glm::vec4(0.f);
		glm::vec4 color = glm::vec4(0.f);

		float t = 0;
		Intersection intersection;
		if(m_pScene->IntersectRayScene(r, &t, &intersection, CPrimitive::FRONT_FACE))
		{
			SceneSample ss(intersection);

			y[0] = m_pScene->GetCamera()->GetCameraAsSceneSample();
			p_A[0] = 1.f;
			y[1] = ss;
			p_A[1] = 1.f; //m_pScene->GetCamera()->GetProbWrtArea(ss);

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
			i++;

			// check implicit path
			if(glm::length(intersection.GetMaterial().emissive) > 0.f)
			{
				float w = 0.f;
				if(i > 1)
				{					
					const float p = ProbA(y[i-1], y[i]);
					w = p / (p_A[i] + p);
				}
				else
				{
					w = 1.f;
				}
				color += w * f_x * intersection.GetMaterial().emissive;
			}

			// make explicit connection
			SceneSample ls_sample;
			m_pScene->SampleLightSource(ls_sample);
			if(m_pScene->Visible(y[i], ls_sample))
			{
				const float w = ls_sample.pdf / (ls_sample.pdf + ProbA(y[i], ls_sample));
				const glm::vec4 L_e = ls_sample.material.emissive;
				const glm::vec4 BRDF = y[i].material.diffuse / PI;
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
				glm::vec3 direction = GetRandomSampleDirectionCosCone(y[i].normal, Rand01(), Rand01(), pdf, 1);
				Ray ray(y[i].position, direction);
				if(m_pScene->IntersectRayScene(ray, &t, &intersection, CPrimitive::FRONT_FACE))
				{
					SceneSample ss(intersection);
					y[i+1] = ss;
					p_A[i+1] = p_A[i] + ProbA(y[i], y[i+1]);

					const glm::vec4 BRDF = y[i].material.diffuse/PI;
					f_x = 1.f/(ProbPSA(y[i], y[i+1]) * t_prob) * f_x * BRDF;
				}
				else
				{
					terminate = true;
				}
			}
		}

		m_pImagePlane->AddSample(pixel, color);
	}
}