#ifndef AVPL_H_
#define AVPL_H_

#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"

class Avpl
{
public:
	Avpl() {}

	Avpl(glm::vec3 position, glm::vec3 normal, glm::vec3 incRadiance, glm::vec3 antiradiance, 
		glm::vec3 incDirection, int bounce, int materialIndex) 
		: 
		m_position(position), m_normal(normal), m_incRadiance(incRadiance), m_antiradiance(antiradiance),
		m_incDirection(incDirection), m_bounce(bounce), m_materialIndex(materialIndex)
	{ 
		glm::vec3 up = glm::vec3(0.0f, 1.0f, 0.0f);
		if(glm::abs(glm::dot(up, m_normal)) > 0.009) {
			up = glm::vec3(0.0f, 0.0f, 1.0f); 
		}

		m_view = glm::lookAt(m_position, m_position + m_normal, up); 

		m_projection = glm::perspective(90.0f, 1.0f, 0.1f, 2000.0f);
	}
	
	~Avpl() {}

	// TODO: remove
	glm::mat4 const& getViewMatrix() const { return m_view; }
	glm::mat4 const& getProjectionMatrix() const { return m_projection; }
	
	glm::vec3 const& getPosition() const { return m_position; }
	glm::vec3 const& getNormal() const { return m_normal; }
	glm::vec3 const& getIncidentDirection() const { return m_incDirection; }
	glm::vec3 const& getIncidentRadiance() const { return m_incRadiance; }
	glm::vec3 const& getAntiradiance() const { return m_antiradiance; }
	
	int getMaterialIndex() const { return m_materialIndex; }
	int getBounce() const { return m_bounce; } 

	void setIncidentRadiance(glm::vec3 const& incRadiance) { m_incRadiance = incRadiance; }
	void setAntiradiance(glm::vec3 const& antiradiance) { m_antiradiance = antiradiance; }
	
private:
	glm::mat4 m_projection;
	glm::mat4 m_view;

	glm::vec3 m_position;
	glm::vec3 m_normal;
	glm::vec3 m_incRadiance;
	glm::vec3 m_incDirection;
	glm::vec3 m_antiradiance;
	int m_bounce;
	int m_materialIndex;
};

#endif // AVPL_H_
