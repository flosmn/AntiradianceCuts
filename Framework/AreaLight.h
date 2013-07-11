#ifndef AREALIGHT_H_
#define AREALIGHT_H_

#include <GL/glew.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/random.hpp>

#include "Utils/Util.h"

#include "Defines.h"

#include "mesh.hpp"
#include "model.hpp"

#include "CMaterialBuffer.h"

#include <vector>
#include <memory>

class AreaLight
{
public:
	AreaLight(float width, float height, glm::vec3 const& centerPosition, glm::vec3 const& frontDirection, glm::vec3 const& upDirection, 
		glm::vec3 const& radiance, CMaterialBuffer* materialBuffer) :
		m_width(width), m_height(height), m_area(width * height), m_centerPosition(centerPosition), m_frontDirection(frontDirection), 
		m_upDirection(upDirection), m_radiance(radiance), m_materialBuffer(materialBuffer), m_materialIndex(0)
	{
		std::vector<glm::vec3> positions;
		positions.push_back(glm::vec3(-1.0f, -1.0f, 0.0f));
		positions.push_back(glm::vec3(-1.0f,  1.0f, 0.0f));
		positions.push_back(glm::vec3( 1.0f,  1.0f, 0.0f));
		positions.push_back(glm::vec3( 1.0f, -1.0f, 0.0f));

		std::vector<glm::vec3> normals;
		for (int i = 0; i < 4; ++i) {
			normals.push_back(glm::vec3(0.f, 0.f, 1.f));
		}
		
		std::vector<glm::uvec3> indices;
		indices.push_back(glm::uvec3(0, 1, 2));
		indices.push_back(glm::uvec3(0, 2, 3));

		MATERIAL m;
		m.emissive = glm::vec4(m_radiance, 1.f);
		m_materialIndex = m_materialBuffer->AddMaterial("LightSource", m);
		
		m_mesh.reset(new Mesh(positions, normals, indices, m_materialIndex));

		updateWorldTransform();
	}

	glm::mat4 const& getWorldTransform() const { return m_worldTransform; }
	glm::vec3 const& getCenterPosition() const { return m_centerPosition; }
	glm::vec3 const& getFrontDirection() const { return m_frontDirection; }
	glm::vec3 const& getRadiance() const { return m_radiance; }
	float getArea() const { return m_area; }
	int getMaterialIndex() const { return m_materialIndex; }
	glm::vec2 getDimensions() const { return glm::vec2(m_width, m_height); }
	Mesh const* getMesh() const { return m_mesh.get(); }

	void setCenterPosition(glm::vec3 const& position) { 
		m_centerPosition = position; 
		updateWorldTransform();
	}

	void setFrontDirection(glm::vec3 const& direction) { 
		m_frontDirection = direction; 
		updateWorldTransform();
	}
	
	void setRadiance(glm::vec3 radiance) { 
		m_radiance = radiance;  
		m_materialBuffer->GetMaterial(m_materialIndex)->emissive =  glm::vec4(m_radiance, 1.f);
	}
	
	glm::vec3 samplePos(float& pdf) const {
		glm::vec2 samplePos = glm::linearRand(glm::vec2(-1, -1), glm::vec2(1, 1));
		glm::vec4 positionTemp = m_worldTransform * glm::vec4(samplePos.x, samplePos.y, 0.0f, 1.0f);		
		glm::vec3 position	= glm::vec3(positionTemp /= positionTemp.w);		
		pdf = 1.f/m_area;
		return position;
	}

	glm::vec3 sampleDir(float& pdf, int order) const {
		glm::vec2 u = glm::linearRand(glm::vec2(-1, -1), glm::vec2(1, 1));
		glm::vec3 direction = GetRandomSampleDirectionCosCone(m_frontDirection, u.x, u.y, pdf, order);
		return direction;
	}

private:
	void updateWorldTransform()	{
		glm::mat4 scale = glm::scale(glm::mat4(), glm::vec3(m_width/2.f, m_height/2.f, 1.0f));
		glm::mat4 position = glm::lookAt(m_centerPosition + 1.f * glm::normalize(m_frontDirection), 
			m_centerPosition - m_frontDirection, m_upDirection);
		position = glm::inverse(position);
		m_worldTransform = position * scale;
	}

	std::unique_ptr<Mesh> m_mesh;
	float m_width;
	float m_height;
	float m_area;
	glm::vec3 m_centerPosition;
	glm::vec3 m_frontDirection;
	glm::vec3 m_upDirection;
	glm::vec3 m_radiance;
	glm::mat4 m_worldTransform;

	CMaterialBuffer* m_materialBuffer;
	int m_materialIndex;
};

#endif
