#ifndef OBJECTCLOUDS_H_
#define OBJECTCLOUDS_H_

#include "OGLResources/COGLVertexArray.h"
#include "OGLResources/COGLUniformBuffer.h"
#include "OGLResources/COGLBindLock.h"
#include "OGLResources/COGLBindSlot.h"
#include "OGLResources/COGLProgram.h"

#include "SimpleObjects.h"

#include "CProgram.h"

#include <memory>
#include <vector>

class PointCloud
{
public:
	PointCloud(std::vector<glm::vec3> const& points, std::vector<glm::vec3> const& colors, COGLUniformBuffer* transform)
	{
		std::shared_ptr<Sphere> sphere = std::make_shared<Sphere>(10);
		m_program.reset(new CProgram("Shaders/PointCloud.vert", "Shaders/PointCloud.frag", "PointCloudProgram"));
		m_program->BindUniformBuffer(transform, "transform");

		m_vertexArray.reset(new COGLVertexArray(GL_TRIANGLES, points.size()));

		m_vertexArray->AddVertexData(0, 3, 
				sphere->getMesh()->getPositions().size() * sizeof(glm::vec3),
				(void*)sphere->getMesh()->getPositions().data());

		m_vertexArray->AddVertexData(1, 3, points.size() * sizeof(glm::vec3), (void*)points.data(), 1);
		m_vertexArray->AddVertexData(2, 3, colors.size() * sizeof(glm::vec3), (void*)colors.data(), 1);

		m_vertexArray->AddIndexData(sphere->getMesh()->getIndices().size() * sizeof(glm::uvec3),
				(void*)sphere->getMesh()->getIndices().data());
		
		m_numIndices = sphere->getMesh()->getIndices().size();

		m_vertexArray->Finish();
	}

	~PointCloud()
	{
	}

	void Draw()
	{
		COGLBindLock lockProgram(m_program->GetGLProgram(), COGL_PROGRAM_SLOT);

		glEnable(GL_DEPTH_TEST);
		glDisable(GL_CULL_FACE);
		m_vertexArray->Draw(3 * m_numIndices);
		glEnable(GL_CULL_FACE);
	}

private:
	std::unique_ptr<COGLVertexArray> m_vertexArray;
	std::unique_ptr<CProgram> m_program;

	int m_numIndices;
};

class AABBCloud
{
public:
	AABBCloud(std::vector<glm::vec3> const& min, std::vector<glm::vec3> const& max, COGLUniformBuffer* transform)
	{
		assert(min.size() == max.size());

		Cube cube;
		m_program.reset(new CProgram("Shaders/AABBCloud.vert", "Shaders/AABBCloud.frag", "AABBCloudProgram"));
		m_program->BindUniformBuffer(transform, "transform");

		m_vertexArray.reset(new COGLVertexArray(GL_TRIANGLES, min.size()));

		m_vertexArray->AddVertexData(0, 3, 
				cube.getMesh()->getPositions().size() * sizeof(glm::vec3),
				(void*)cube.getMesh()->getPositions().data());

		m_vertexArray->AddVertexData(1, 3, min.size() * sizeof(glm::vec3), (void*)min.data(), 1);
		m_vertexArray->AddVertexData(2, 3, max.size() * sizeof(glm::vec3), (void*)max.data(), 1);

		m_vertexArray->AddIndexData(cube.getMesh()->getIndices().size() * sizeof(glm::uvec3),
				(void*)cube.getMesh()->getIndices().data());
		
		m_numIndices = cube.getMesh()->getIndices().size();

		m_vertexArray->Finish();
	}

	~AABBCloud()
	{
	}

	void Draw()
	{
		COGLBindLock lockProgram(m_program->GetGLProgram(), COGL_PROGRAM_SLOT);

		glEnable(GL_DEPTH_TEST);
		glDisable(GL_CULL_FACE);

		glEnable(GL_POLYGON_OFFSET_FILL);
		glPolygonOffset(1.1f, 4.0f);

		glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
		glLineWidth(5.f);
		m_vertexArray->Draw(3 * m_numIndices);
		glLineWidth(1.f);
		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
		glEnable(GL_CULL_FACE);
		
		glDisable(GL_POLYGON_OFFSET_FILL);
	}

private:
	std::unique_ptr<COGLVertexArray> m_vertexArray;
	std::unique_ptr<CProgram> m_program;

	int m_numIndices;
};

#endif // OBJECTCLOUDS_H_
