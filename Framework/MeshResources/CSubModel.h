#ifndef _C_SUB_MODEL_H_
#define _C_SUB_MODEL_H_

typedef unsigned int uint;

#include "glm/glm.hpp"

#include "..\Structs.h"

#include <vector>

#include "CModel.h"

class Triangle;

class COGLVertexArray;
class COGLVertexBuffer;
class COGLUniformBuffer;
class CMeshGeometry;
class CMeshMaterial;
class CMesh;

class CSubModel : public CModel
{
public:
	CSubModel();
	~CSubModel();

	bool Init(CMesh* mesh);
	bool Init(CMeshGeometry* pMeshGeometry);
	void Release();
	
	void Draw(COGLUniformBuffer* pUBMaterial);
	void Draw();
	
	virtual void SetMaterial(MATERIAL& mat);
	virtual MATERIAL& GetMaterial();
		
	std::vector<Triangle*> GetTriangles();

private:	
	COGLVertexArray* m_pGLVARenderData;
	CMeshMaterial* m_pMaterial;
	
	std::vector<Triangle*> m_Triangles;
	uint m_nTriangles;
};

#endif _C_SUB_MODEL_H_