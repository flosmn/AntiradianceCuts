#ifndef MESH_HPP_
#define MESH_HPP_

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
			m_material(-1)
		{
			m_vertexArray.reset(new COGLVertexArray(GL_TRIANGLES));
			m_vertexArray->AddVertexData(0, 3, m_positions.size() * sizeof(glm::vec3), m_positions.data());
			m_vertexArray->AddVertexData(1, 3, m_normals.size() * sizeof(glm::vec3), m_normals.data());
			m_vertexArray->AddIndexData(m_indices.size() * sizeof(glm::uvec3), m_indices.data());
			m_vertexArray->Finish();
		}

		Mesh(std::vector<glm::vec3> const& positions, 
			std::vector<glm::vec3> const& normals, 
			std::vector<glm::uvec3> const& indices,
			int material) :
			m_positions(positions),
			m_normals(normals),
			m_indices(indices),
			m_material(material)
		{
			std::vector<int> m_materials(m_positions.size());
			fill(m_materials.begin(), m_materials.end(), m_material);

			m_vertexArray.reset(new COGLVertexArray(GL_TRIANGLES));
			m_vertexArray->AddVertexData(0, 3, m_positions.size() * sizeof(glm::vec3), m_positions.data());
			m_vertexArray->AddVertexData(1, 3, m_normals.size() * sizeof(glm::vec3), m_normals.data());
			m_vertexArray->AddVertexData(2, 1, m_materials.size() * sizeof(int), m_materials.data());
			m_vertexArray->AddIndexData(m_indices.size() * sizeof(glm::uvec3), m_indices.data());
			m_vertexArray->Finish();
		}

		std::vector<glm::vec3> const& getPositions() const { return m_positions; }
		std::vector<glm::uvec3> const& getIndices() const { return m_indices; }
		std::vector<glm::vec3> const& getNormals() const { return m_normals; }
		int getMaterial() const { return m_material; }

		bool hasNormals() const { return m_normals.size() != 0; }
		bool hasMaterial() const { return m_material != -1; }

		void draw() const {
			m_vertexArray->Draw(3 * m_indices.size());
		}

	private:
		std::unique_ptr<COGLVertexArray> m_vertexArray;
		int m_material;
		
		std::vector<glm::vec3> m_positions;
		std::vector<glm::vec3> m_normals;
		std::vector<glm::uvec3> m_indices;
};

#endif // MESH_HPP_
