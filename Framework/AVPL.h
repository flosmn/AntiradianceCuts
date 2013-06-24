#ifndef _AVPL__H
#define _AVPL__H

#include "glm/glm.hpp"

#include "Structs.h"

#include "Intersection.h"

class CConfigManager;
class CMaterialBuffer;

class AVPL
{
public:
	AVPL();
	AVPL(glm::vec3 p, glm::vec3 n, glm::vec3 L_in, glm::vec3 A, glm::vec3 w, float m_ConeAngle, int bounce, int materialIndex, 
		CMaterialBuffer* pMaterialBuffer, CConfigManager* pConfManager);
	~AVPL();

	glm::mat4 GetViewMatrix() const;
	glm::mat4 GetProjectionMatrix() const;

	glm::vec3 GetPosition() const { return m_Position; }
	glm::vec3 GetOrientation() const { return m_Orientation; }
	
	glm::vec3 GetRadiance(const glm::vec3& w) const;
	glm::vec3 GetAntiradiance(const glm::vec3& w) const;

	glm::vec3 GetIncidentRadiance() const { return m_Radiance; }

	glm::vec3 GetDirection() const { return m_Direction; }
	float GetConeAngle() const { return m_ConeAngle; }
	int GetMaterialIndex() const { return m_MaterialIndex; }
	
	void ScaleIncidentRadiance(float s) { m_Radiance *= s; }
	void ScaleAntiradiance(float s) { m_Antiradiance *= s; }
	
	glm::vec3 GetIrradiance(const SceneSample& ss) const;
	glm::vec3 GetAntiirradiance(const SceneSample& ss) const;

	glm::vec3 SampleAntiradianceDirection();

	int GetBounce() const { return m_Bounce; } 
	
	void Fill(AVPL_STRUCT& avpl) const;
	void Fill(AVPL_BUFFER& avpl_buffer) const;

	void SetColor(glm::vec3 c) { m_DebugColor = c; }
	glm::vec3 GetColor() const { return m_DebugColor; }

public:
	glm::vec3 GetAntiradiance(const glm::vec3& w, const float angleFactor) const;

	glm::vec3 m_Position;
	glm::vec3 m_Orientation;
	glm::vec3 m_Radiance;
	glm::vec3 m_Antiradiance;
	glm::vec3 m_Direction;
	glm::vec3 m_DebugColor;
	float m_ConeAngle;
	int m_Bounce;
	int m_MaterialIndex;
	int padd;

	CConfigManager* m_confManager;
	CMaterialBuffer* m_pMaterialBuffer;
};

#endif
