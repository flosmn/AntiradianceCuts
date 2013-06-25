#ifndef POINTCLOUD_H_
#define POINTCLOUD_H_

#include "OGLResources/COGLVertexArray.h"

#include "Sphere.h"

#include <memory>
#include <vector>

class PointCloud
{
public:
	PointCloud(std::vector<glm::vec3> const& points, std::vector<glm::vec3> const& colors)
	{
		std::shared_ptr<Sphere> sphere = std::make_shared<Sphere>(10);

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
		m_vertexArray->Draw(3 * m_numIndices);
	}

private:
	std::unique_ptr<COGLVertexArray> m_vertexArray;
	int m_numIndices;
};

#endif // POINTCLOUD_H_
