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
class CMeshMaterial;

class CTriangle;

typedef unsigned int uint;
typedef unsigned short ushort;

class CSubModel : public CModel
{
public:
	CSubModel();
	~CSubModel();

	bool Init(CMesh* pMesh);
	bool Init(CMeshGeometry* pMeshGeometry);
	void Release();
	
	void Draw(COGLUniformBuffer* pUBMaterial);
	void Draw();
	
	virtual void SetMaterial(const MATERIAL& mat);
	virtual MATERIAL GetMaterial() const;

	void SetWorldTransform(const glm::mat4& transform);
		
	const std::vector<CTriangle*>& GetTrianglesOS() const;
	const std::vector<CTriangle*>& GetTrianglesWS() const;

private:	
	void CreateTriangleData(uint nFaces, const ushort* pIndexData, 
		const glm::vec4* pPositionData, const glm::vec3* pNormalData);
	
	COGLVertexArray* m_pGLVARenderData;
	CMeshMaterial* m_pMaterial;
	
	std::vector<CTriangle*> m_TrianglesOS;
	std::vector<CTriangle*> m_TrianglesWS;

	uint m_nTriangles;
};

#endif _C_SUB_MODEL_H_