#ifndef _C_MODEL_H
#define _C_MODEL_H

typedef unsigned int uint;

#include "GL/glew.h"
#include "glm/glm.hpp"

#include "..\Structs.h"

#include <vector>

class Triangle;

class CMesh;
class CSubModel;
class CMeshGeometry;
class CMeshMaterial;
class CGLUniformBuffer;

class CModel 
{
public:
	CModel();
	~CModel();

	bool Init(CMesh* m);
	bool Init(std::string name);
	void Release();
	
	void Draw(const glm::mat4& mView, const glm::mat4& mProj, CGLUniformBuffer* pUBTransform, CGLUniformBuffer* pUBMaterial);
	void Draw(const glm::mat4& mView, const glm::mat4& mProj, CGLUniformBuffer* pUBTransform);
	
	void SetWorldTransform(glm::mat4 matrix) { worldTransform = matrix; }
	glm::mat4 GetWorldTransform() { return worldTransform; }
	
	virtual void SetMaterial(MATERIAL& mat);	
	virtual MATERIAL& GetMaterial();
		
	std::vector<CSubModel*> GetSubModels() { return m_vecSubModels; }

private:	
	CMesh* m_Mesh;
	
	uint m_nTriangles;

	std::vector<CMeshGeometry*> m_MeshGeometries;
	std::vector<CMeshMaterial*> m_MeshMaterials;

	std::vector<CSubModel*> m_vecSubModels;
		
	glm::mat4 worldTransform;		
};

#endif