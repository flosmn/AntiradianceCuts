#ifndef _C_FULL_SCREEN_QUAD_H_
#define _C_FULL_SCREEN_QUAD_H_

#include "CMesh.h"

#include "..\OGLResources\COGLVertexArray.h"

class CFullScreenQuad
{
public:
	CFullScreenQuad() {
		m_mesh.reset(new CFullScreenQuadMesh());
		m_vertexArray.reset(new COGLVertexArray(GL_TRIANGLES, "VertexArray"));

		GLuint positionDataSize		= sizeof(glm::vec4) * m_mesh->GetNumberOfVertices();
		GLuint textureDataSize		= sizeof(glm::vec3) * m_mesh->GetNumberOfVertices();
		GLuint indexDataSize		= sizeof(GLuint)*3  * m_mesh->GetNumberOfFaces();

		m_vertexArray->AddVertexData(0, 4, positionDataSize, (void*)m_mesh->GetPositionData());
		m_vertexArray->AddVertexData(1, 3, textureDataSize, (void*)m_mesh->GetVertexTexCoords());
		m_vertexArray->AddIndexData(indexDataSize, (void*)m_mesh->GetIndexData());

		m_vertexArray->Finish();
	}

	~CFullScreenQuad() {
	}

	void Draw() {
		m_vertexArray->Draw(3 * m_mesh->GetNumberOfFaces());
	}

private:
	std::unique_ptr<COGLVertexArray> m_vertexArray;
	std::unique_ptr<CMesh> m_mesh;
};

#endif // _C_FULL_SCREEN_QUAD_H_
