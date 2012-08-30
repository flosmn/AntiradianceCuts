#ifndef _C_MODEL_H
#define _C_MODEL_H

typedef unsigned int uint;

#include "GL/glew.h"
#include "glm/glm.hpp"

#include "..\Structs.h"

#include <vector>

class CMesh;
class CSubModel;
class CMeshGeometry;
class CMeshMaterial;
class COGLUniformBuffer;
class COGLVertexArray;
class CTriangle;
class CMaterialBuffer;

class CModel 
{
public:
	CModel();
	~CModel();

	bool Init(CMesh* m);
	bool Init(std::string name, CMaterialBuffer* pMaterialBuffer);
	void Release();
	
	void Draw(const glm::mat4& mView, const glm::mat4& mProj, COGLUniformBuffer* pUBTransform, COGLUniformBuffer* pUBMaterial);
	void Draw(const glm::mat4& mView, const glm::mat4& mProj, COGLUniformBuffer* pUBTransform);
	
	void SetWorldTransform(const glm::mat4& matrix);
	glm::mat4 GetWorldTransform() const { return m_WorldTransform; }
	
	glm::vec3 GetPositionWS();
	std::vector<CSubModel*>& GetSubModels() { return m_SubModels; }
protected:
	glm::mat4 m_WorldTransform;

private:	
	CMesh* m_Mesh;
	
	std::vector<CSubModel*> m_SubModels;

	uint m_nTriangles;
};

#endif