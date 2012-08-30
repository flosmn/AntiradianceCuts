#ifndef _C_SUB_MODEL_H_
#define _C_SUB_MODEL_H_

typedef unsigned int uint;

#include "glm/glm.hpp"

#include "..\Structs.h"

#include <vector>

#include <glm/glm.hpp>

#include "CModel.h"

class COGLVertexArray;
class COGLVertexBuffer;
class COGLUniformBuffer;
class CMeshGeometry;

class CTriangle;
class CMaterialBuffer;

typedef unsigned int uint;
typedef unsigned int uint;

class CSubModel : public CModel
{
public:
	CSubModel();
	~CSubModel();

	bool Init(CMesh* pMesh);
	bool Init(const CMeshGeometry& meshGeometry, CMaterialBuffer* pMaterialBuffer);
	void Release();
	
	void Draw(COGLUniformBuffer* pUBMaterial);
	void Draw();
		
	void SetWorldTransform(const glm::mat4& transform);
		
	const std::vector<CTriangle*>& GetTrianglesOS() const;
	const std::vector<CTriangle*>& GetTrianglesWS() const;

	uint GetMaterialIndex() { return m_MaterialIndex; }

private:	
	void CreateTriangleData(uint nFaces, const uint* pIndexData, 
		const glm::vec4* pPositionData, const glm::vec3* pNormalData);
	
	COGLVertexArray* m_pGLVARenderData;
	
		
	std::vector<CTriangle*> m_TrianglesOS;
	std::vector<CTriangle*> m_TrianglesWS;

	uint m_MaterialIndex;
	uint m_nTriangles;

	CMaterialBuffer* m_pMaterialBuffer;
};

#endif _C_SUB_MODEL_H_