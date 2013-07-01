#ifndef _C_SUB_MODEL_H_
#define _C_SUB_MODEL_H_

typedef unsigned int uint;

#include "glm/glm.hpp"

#include "..\Structs.h"

#include <vector>
#include <memory>

#include <glm/glm.hpp>

#include "CModel.h"

class COGLVertexArray;
class COGLVertexBuffer;
class COGLUniformBuffer;
class CMeshGeometry;

class Triangle;
class CMaterialBuffer;

typedef unsigned int uint;
typedef unsigned int uint;

class CSubModel
{
public:
	CSubModel(CMesh* pMesh);
	CSubModel(const CMeshGeometry& meshGeometry, CMaterialBuffer* materialBuffer);
	virtual ~CSubModel();

	void Draw(COGLUniformBuffer* ubMaterial);
	void Draw();
		
	void SetWorldTransform(const glm::mat4& transform);
		
	std::vector<Triangle>& GetTrianglesOS();
	std::vector<Triangle>& GetTrianglesWS();

	uint GetMaterialIndex() { return m_MaterialIndex; }

private:	
	void CreateTriangleData(uint nFaces, const uint* pIndexData, 
		const glm::vec4* pPositionData, const glm::vec3* pNormalData);
	
	CMaterialBuffer* m_materialBuffer;

	std::unique_ptr<COGLVertexArray> m_vertexArray;
		
	std::vector<Triangle> m_trianglesOS;
	std::vector<Triangle> m_trianglesWS;

	uint m_MaterialIndex;
	uint m_nTriangles;
	glm::mat4 m_worldTransform;
};

#endif _C_SUB_MODEL_H_
