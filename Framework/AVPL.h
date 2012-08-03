#ifndef _AVPL__H
#define _AVPL__H

#include "glm/glm.hpp"

#include "Structs.h"

class CConfigManager;

class AVPL
{
public:
	AVPL();
	AVPL(glm::vec3 p, glm::vec3 n, glm::vec3 I, glm::vec3 A, glm::vec3 w_A, int bounce, CConfigManager* pConfManager);
	~AVPL();

	glm::mat4 GetViewMatrix() const;
	glm::mat4 GetProjectionMatrix() const;

	glm::vec3 GetPosition() const { return m_Position; }
	glm::vec3 GetOrientation() const { return m_Orientation; }
	glm::vec3 GetIntensity(glm::vec3 w) const;
	glm::vec3 GetAntiintensity(glm::vec3 w, const float N) const;
	glm::vec3 GetAntiradianceDirection() const { return m_AntiradianceDirection; }
	glm::vec3 GetMaxAntiintensity() const { return m_Antiintensity; }
	glm::vec3 GetMaxIntensity() const { return m_Intensity; }
	
	void SetIntensity(glm::vec3 i) { m_Intensity = i; }
	void SetAntiintensity(glm::vec3 a) { m_Antiintensity = a; }
	
	glm::vec3 GetIrradiance(const SceneSample& ss) const;
	glm::vec3 GetAntiirradiance(const SceneSample& ss, const float N) const;

	glm::vec3 SampleAntiradianceDirection(const float N);

	int GetBounce() const { return m_Bounce; } 
	
	void Fill(AVPL_STRUCT& avpl) const;
	void Fill(AVPL_BUFFER& avpl_buffer) const;

	void SetColor(glm::vec3 c) { m_DebugColor = c; }
	glm::vec3 GetColor() const { return m_DebugColor; }

private:
	glm::vec3 m_Position;
	glm::vec3 m_Orientation;
	glm::vec3 m_Intensity;
	glm::vec3 m_Antiintensity;
	glm::vec3 m_AntiradianceDirection;
	glm::vec3 m_DebugColor;
	int m_Bounce;

	CConfigManager* m_pConfManager;
};

#endif