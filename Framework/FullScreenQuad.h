#ifndef FULLSCREENQUAD_H_
#define FULLSCREENQUAD_H_

#include <glm/glm.hpp>

#include "OGLResources\COGLVertexArray.h"

#include <memory>
#include <vector>

class FullScreenQuad
{
public:
	FullScreenQuad() {
		std::vector<glm::vec3> positions;
		positions.push_back(glm::vec3(-1.0f, -1.0f, -1.0f));
		positions.push_back(glm::vec3(-1.0f,  1.0f, -1.0f));
		positions.push_back(glm::vec3( 1.0f,  1.0f, -1.0f));
		positions.push_back(glm::vec3( 1.0f, -1.0f, -1.0f));

		std::vector<glm::vec2> texCoords;
		texCoords.push_back(glm::vec2(0.0f, 0.0f));
		texCoords.push_back(glm::vec2(0.0f, 1.0f));
		texCoords.push_back(glm::vec2(1.0f, 1.0f));
		texCoords.push_back(glm::vec2(1.0f, 0.0f));

		std::vector<glm::uvec3> indices;
		indices.push_back(glm::uvec3(0, 1, 2));
		indices.push_back(glm::uvec3(0, 2, 3));
		
		m_vertexArray.reset(new COGLVertexArray(GL_TRIANGLES, "VertexArray"));
		
		m_vertexArray->AddVertexData(0, 3, positions.size() * sizeof(glm::vec3), (void*)positions.data());
		m_vertexArray->AddVertexData(1, 2, texCoords.size() * sizeof(glm::vec2), (void*)texCoords.data());
		m_vertexArray->AddIndexData(indices.size() * sizeof(glm::uvec3), (void*)indices.data());

		m_vertexArray->Finish();
	}
	
	void draw() {
		m_vertexArray->Draw(6);
	}

private:
	std::unique_ptr<COGLVertexArray> m_vertexArray;
};

#endif // FULLSCREENQUAD_H_
