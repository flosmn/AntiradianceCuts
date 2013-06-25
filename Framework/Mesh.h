#ifndef _MESH_H_
#define _MESH_H_

#include "OGLResources/COGLVertexArray.h"

#include <string>
#include <vector>

#include <glm/glm.hpp>

class Mesh
{
	public:
		Mesh(std::vector<glm::vec3> const& positions, 
			std::vector<glm::vec3> const& normals, 
			std::vector<glm::uvec3> const& indices) :
			m_positions(positions),
			m_normals(normals),
			m_indices(indices),
			m_numIndices(indices.size())
		{
			m_vertexArray.reset(new COGLVertexArray(GL_TRIANGLES));
			m_vertexArray->AddVertexData(0, 3, m_positions.size() * sizeof(glm::vec3), m_positions.data());
			//m_vertexArray->AddVertexData(1, 3, m_normals.size() * sizeof(glm::vec3), m_normals.data());
			m_vertexArray->AddIndexData(m_indices.size() * sizeof(glm::uvec3), m_indices.data());
			m_vertexArray->Finish();
		}

		std::vector<glm::vec3> const& getPositions() { return m_positions; }
		std::vector<glm::uvec3> const& getIndices() { return m_indices; }
		std::vector<glm::vec3> const& getNormals() { return m_normals; }

		bool hasNormals() { return m_normals.size() != 0; }

		void draw() {
			m_vertexArray->Draw(3 * m_numIndices);
		}

	private:
		std::unique_ptr<COGLVertexArray> m_vertexArray;
		int m_numIndices;
		
		std::vector<glm::vec3> m_positions;
		std::vector<glm::uvec3> m_indices;
		std::vector<glm::vec3> m_normals;
};

#endif // _MESH_H_
