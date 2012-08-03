#include "AVPL.h"

#include <glm/gtx/transform.hpp>

#include "CConfigManager.h"

#include "Utils\Util.h"
#include "Utils\Rand.h"

static glm::mat4 projMatrix = glm::perspective(90.0f, 1.0f, 0.1f, 100.0f);

AVPL::AVPL(glm::vec3 p, glm::vec3 n, glm::vec3 I,
	glm::vec3 A, glm::vec3 w_A, int bounce, CConfigManager* pConfManager)
	: m_pConfManager(pConfManager)
{
	m_Position = p;
	m_Orientation = n;
	m_Intensity = I;
	m_Antiintensity= A;
	m_AntiradianceDirection = w_A;
	m_Bounce = bounce;
	m_DebugColor = glm::vec3(0.2f, 0.2f, 0.2f);
}

glm::mat4 AVPL::GetViewMatrix() const 
{
	glm::vec3 up = glm::vec3(0.0f, 1.0f, 0.0f);
	if(glm::abs(glm::dot(up, m_Orientation)) > 0.009) {
		up = glm::vec3(0.0f, 0.0f, 1.0f); 
	}

	return glm::lookAt(m_Position, m_Position + m_Orientation, up); 
}

glm::mat4 AVPL::GetProjectionMatrix() const {
	return glm::perspective(90.0f, 1.0f, 0.1f, 100.0f);
}
 
AVPL::~AVPL()
{
}

AVPL::AVPL() { }

void AVPL::Fill(AVPL_STRUCT& avpl) const
{
	avpl.pos = glm::vec4(m_Position, 1.f);
	avpl.norm = glm::vec4(m_Orientation, 1.f);
	avpl.I = glm::vec4(m_Intensity, 1.f);
	avpl.A = glm::vec4(m_Antiintensity, 1.f);
	avpl.w_A = m_AntiradianceDirection;
	avpl.DebugColor = glm::vec4(m_DebugColor, 1.f);
	avpl.ViewMatrix = GetViewMatrix();
	avpl.ProjectionMatrix = GetProjectionMatrix();
	avpl.Bounce = m_Bounce;
}

void AVPL::Fill(AVPL_BUFFER& avpl) const
{
	memset(&avpl, 0, sizeof(AVPL_BUFFER));
	avpl.pos = glm::vec4(m_Position, 1.f);
	avpl.norm = glm::vec4(m_Orientation, 0.f);
	avpl.I = glm::vec4(m_Intensity, 0.f);
	avpl.A = glm::vec4(m_Antiintensity, 0.f);
	avpl.w_A = glm::vec4(m_AntiradianceDirection, 0.f);
}

glm::vec3 AVPL::GetIntensity(glm::vec3 w) const
{
	return clamp(glm::dot(w, m_Orientation), 0, 1) * m_Intensity;
}

glm::vec3 AVPL::GetAntiintensity(glm::vec3 w, const float N) const
{
	const float N_OVER_PI = N / PI;
	glm::vec3 res = glm::vec3(0.f);

	if(glm::dot(w, m_AntiradianceDirection) < 0.01f)
	{
		return res;
	}

	const float theta = acos(clamp(glm::dot(w, m_AntiradianceDirection), 0, 1));

	if(theta < PI/N)
	{
		res = m_Antiintensity;

		if(m_pConfManager->GetConfVars()->AntiradFilterMode == 1)
		{
			res = - 2.f * res * m_pConfManager->GetConfVars()->AntiradFilterK * (N_OVER_PI * theta - 1);
		}
		if(m_pConfManager->GetConfVars()->AntiradFilterMode == 2)
		{
			const float M = m_pConfManager->GetConfVars()->AntiradFilterGaussFactor;
			const float s = PI / (M*N);
			res = m_pConfManager->GetConfVars()->AntiradFilterK * ( exp(-(theta*theta)/(s*s)) - exp(-(M*M)) ) * res;
		}
	}

	return res;
}

glm::vec3 AVPL::GetIrradiance(const SceneSample& ss) const
{
	return G(m_Position, m_Orientation, ss.position, ss.normal) * GetIntensity(glm::normalize(ss.position - m_Position));
}

glm::vec3 AVPL::GetAntiirradiance(const SceneSample& ss, const float N) const
{
	const glm::vec3 w = glm::normalize(ss.position - m_Position);
	return G_A(m_Position, m_Orientation, ss.position, ss.normal) * GetAntiintensity(w, N);
}

glm::vec3 AVPL::SampleAntiradianceDirection(const float N)
{
	float pdf = 0.f;
	glm::vec3 dir = SampleConeDirection(m_AntiradianceDirection, PI/N, Rand01(), Rand01(), &pdf);
	return dir;
}