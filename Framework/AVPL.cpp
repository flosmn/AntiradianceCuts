#include "AVPL.h"

#include <glm/gtx/transform.hpp>

#include "Utils\Util.h"
#include "Utils\Rand.h"

AVPL::AVPL(glm::vec3 p, glm::vec3 n, glm::vec3 I,
	glm::vec3 A, glm::vec3 w_A, int bounce)
{
	m_Position = p;
	m_Orientation = n;
	m_Intensity = I;
	m_Antiintensity= A;
	m_AntiradianceDirection = w_A;
	m_Bounce = bounce;
	
	m_DebugColor = glm::vec3(0.8f, 0.8f, 0.0f);

	m_ProjectionMatrix = glm::perspective(90.0f, 1.0f, 0.1f, 100.0f);

	glm::vec3 up = glm::vec3(0.0f, 1.0f, 0.0f);
	if(glm::abs(glm::dot(up, m_Orientation)) > 0.009) {
		up = glm::vec3(0.0f, 0.0f, 1.0f); 
	}

	m_ViewMatrix = glm::lookAt(m_Position, m_Position + m_Orientation, up);
}
 
AVPL::~AVPL()
{
}

AVPL::AVPL() { }

void AVPL::Fill(AVPL_STRUCT& avpl)
{
	avpl.pos = glm::vec4(m_Position, 1.f);
	avpl.norm = glm::vec4(m_Orientation, 1.f);
	avpl.I = glm::vec4(m_Intensity, 1.f);
	avpl.A = glm::vec4(m_Antiintensity, 1.f);
	avpl.w_A = m_AntiradianceDirection;
	avpl.DebugColor = glm::vec4(m_DebugColor, 1.f);
	avpl.ViewMatrix = m_ViewMatrix;
	avpl.ProjectionMatrix = m_ProjectionMatrix;
	avpl.Bounce = m_Bounce;
}

void AVPL::Fill(AVPL_BUFFER& avpl)
{
	memset(&avpl, 0, sizeof(AVPL_BUFFER));
	avpl.pos = glm::vec4(m_Position, 1.f);
	avpl.norm = glm::vec4(m_Orientation, 1.f);
	avpl.I = glm::vec4(m_Intensity, 1.f);
	avpl.A = glm::vec4(m_Antiintensity, 1.f);
	avpl.w_A = m_AntiradianceDirection;
	avpl.Bounce = m_Bounce;
}

glm::vec3 AVPL::GetIntensity(glm::vec3 w)
{
	return clamp(glm::dot(w, m_Orientation), 0, 1) * m_Intensity;
}

glm::vec3 AVPL::GetAntiintensity(glm::vec3 w, const float& N)
{
	glm::vec3 res = glm::vec3(0.f);

	if(glm::dot(w, m_AntiradianceDirection) < 0.01f)
	{
		return res;
	}

	const float theta = acos(clamp(glm::dot(w, m_AntiradianceDirection), 0, 1));

	if(theta < PI/N)
	{
		//const float K = (PI * (1 - cos(PI/N))) / (PI - N * sin(PI/N));
		//res = K * (1 - theta / PI/N) * m_Antiintensity;
		res = m_Antiintensity;
	}

	return res;
}

glm::vec3 AVPL::SampleAntiradianceDirection(const float& N)
{
	float pdf = 0.f;
	glm::vec3 dir = SampleConeDirection(m_AntiradianceDirection, PI/N, Rand01(), Rand01(), &pdf);
	return dir;
}