#ifndef _C_POINT_CLOUD_H_
#define _C_POINT_CLOUD_H_

#include "glm/glm.hpp"

#include <vector>

typedef unsigned int uint;

#include "OGLResources\COGLVertexArray.h"

class CPointCloud
{
public:
	CPointCloud() {};
	~CPointCloud() {};

	void Draw()
	{
		m_numPoints = (uint)positions.size();
		
		m_vertexArray.reset(new COGLVertexArray(GL_POINTS, "vertexArray"));

		m_vertexArray->AddVertexData(0, (uint)positions.size(), (uint)positions.size() * sizeof(glm::vec4), (void*)positions.data());
		m_vertexArray->AddVertexData(1, (uint)colors.size(), (uint)colors.size() * sizeof(glm::vec4), (void*)colors.data());

		std::vector<GLuint> indexData;
		for (size_t i = 0; i < positions.size(); ++i) {
			indexData.push_back(GLuint(i));
		}

		m_vertexArray->AddIndexData((uint)indexData.size() * sizeof(GLuint), indexData.data());
		m_vertexArray->Finish();

		glPointSize(3.0f);
		m_vertexArray->Draw(m_numPoints);
	}

	std::vector<glm::vec4> positions;
	std::vector<glm::vec4> colors;
private:
	std::unique_ptr<COGLVertexArray> m_vertexArray;
	uint m_numPoints;
};

#endif // _C_POINT_CLOUD_H_
