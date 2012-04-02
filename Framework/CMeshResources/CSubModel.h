#ifndef _C_SUB_MODEL_H_
#define _C_SUB_MODEL_H_

typedef unsigned int uint;

#include "glm/glm.hpp"

#include "..\Structs.h"

#include <vector>

class Triangle;

class CGLVertexArray;
class CGLVertexBuffer;
class CGLUniformBuffer;
class CMeshGeometry;
class CMeshMaterial;
class CMesh;

class CSubModel 
{
public:
	CSubModel();
	~CSubModel();

	bool Init(CMesh* mesh);
	bool Init(CMeshGeometry* pMeshGeometry);
	void Release();
	
	void Draw(CGLUniformBuffer* pUBMaterial);
	void Draw();
	
	void SetMaterial(MATERIAL& mat);
	MATERIAL& GetMaterial();
		
	std::vector<Triangle*> GetTriangles();

private:	
	CGLVertexArray* m_pGLVARenderData;
	CMeshMaterial* m_pMaterial;
		
	uint m_nTriangles;
};

#endif _C_SUB_MODEL_H_