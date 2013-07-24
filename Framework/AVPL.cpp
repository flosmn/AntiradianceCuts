#include "AVPL.h"

#include <iostream>

#include <glm/gtx/transform.hpp>

#include "CConfigManager.h"
#include "CMaterialBuffer.h"

#include "Utils\Util.h"
#include "Utils\Rand.h"

#define EPSILON 0.001f

static glm::mat4 projMatrix = glm::perspective(90.0f, 1.0f, 0.1f, 100.0f);

AVPL::AVPL(glm::vec3 p, glm::vec3 n, glm::vec3 L,
	glm::vec3 A, glm::vec3 w, 
	int bounce, int materialIndex, 
	CMaterialBuffer* pMaterialBuffer, CConfigManager* pConfManager)
	: m_pMaterialBuffer(pMaterialBuffer), m_confManager(pConfManager)
{
	m_Position = p;
	m_Orientation = n;
	m_Radiance = L;
	m_Antiradiance= A;
	m_Direction = w;
	m_ConeAngle = M_PI / 10.f;
	m_Bounce = bounce;
	m_MaterialIndex = materialIndex;
	m_DebugColor = glm::vec3(0.2f, 0.2f, 0.2f);

	if (glm::dot(m_Orientation, -m_Direction) < 0.f)
		std::cout << "wrong" << std::endl;
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
	return glm::perspective(90.0f, 1.0f, 0.1f, 2000.0f);
}
 
AVPL::~AVPL()
{
}

AVPL::AVPL() { }

void AVPL::Fill(AVPL_STRUCT& avpl) const
{
	memset(&avpl, 0, sizeof(AVPL_STRUCT));
	avpl.pos = glm::vec4(m_Position, 1.f);
	avpl.norm = glm::vec4(m_Orientation, 0.f);
	avpl.L = glm::vec4(m_Radiance, 0.f);
	avpl.A = glm::vec4(m_Antiradiance, 0.f);
	avpl.w = glm::vec4(m_Direction, 0.f);
	avpl.DebugColor = glm::vec4(m_DebugColor, 1.f);
	avpl.angleFactor = m_ConeAngle;	
	avpl.bounce = float(m_Bounce);
	avpl.materialIndex = float(m_MaterialIndex);
	avpl.ViewMatrix = GetViewMatrix();
	avpl.ProjectionMatrix = GetProjectionMatrix();
}

void AVPL::Fill(AVPL_BUFFER& avpl) const
{
	memset(&avpl, 0, sizeof(AVPL_BUFFER));
	avpl.pos = glm::vec4(m_Position, 1.f);
	avpl.norm = glm::vec4(m_Orientation, 0.f);
	avpl.L = glm::vec4(m_Radiance, 0.f);
	avpl.A = glm::vec4(m_Antiradiance, 0.f);
	avpl.w = glm::vec4(m_Direction, 0.f);
	avpl.angleFactor = m_ConeAngle;
	avpl.materialIndex = float(m_MaterialIndex);
}

glm::vec3 AVPL::GetAntiradiance(const glm::vec3& w) const
{
	return m_Antiradiance;
}

glm::vec3 AVPL::GetRadiance(const glm::vec3& w) const
{
	MATERIAL* mat = m_pMaterialBuffer->GetMaterial(m_MaterialIndex);
	if(glm::length(GetDirection()) == 0.f)
		return m_Radiance;

	glm::vec3 BRDF = glm::vec3(Phong(-GetDirection(), w, GetOrientation(), mat));
	return BRDF * m_Radiance;
}

glm::vec3 AVPL::GetIrradiance(const SceneSample& ss) const
{
	glm::vec3 direction = glm::normalize(ss.position - m_Position);
	return G(m_Position, m_Orientation, ss.position, ss.normal) * GetRadiance(direction);
}

glm::vec3 AVPL::GetAntiirradiance(const SceneSample& ss) const
{
	const glm::vec3 w = glm::normalize(ss.position - m_Position);
	return G_A(m_Position, m_Orientation, ss.position, ss.normal) * GetAntiradiance(w);
}

glm::vec3 AVPL::SampleAntiradianceDirection()
{
	float pdf = 0.f;
	glm::vec3 dir = SampleConeDirection(m_Direction, M_PI/m_ConeAngle, Rand01(), Rand01(), &pdf);
	return dir;
}
