#include "CModel.h"

#include "..\Macros.h"
#include "..\Structs.h"
#include "..\Camera.h"

#include "CMesh.h"
#include "CSubModel.h"
#include "CMeshGeometry.h"
#include "CMeshMaterial.h"
#include "CObjFileLoader.h"

#include "..\Utils\GLErrorUtil.h"

#include "..\OGLResources\COGLVertexBuffer.h"
#include "..\OGLResources\COGLVertexArray.h"
#include "..\OGLResources\COGLBindLock.h"
#include "..\OGLResources\COGLUniformBuffer.h"


CModel::CModel()
	: m_Mesh(nullptr)
{

}

CModel::~CModel()
{
	std::vector<CSubModel*>::iterator it;
	for(it = m_vecSubModels.begin(); it < m_vecSubModels.end(); ++it)
	{
		SAFE_DELETE(*it);
	}
	m_vecSubModels.clear();
};

bool CModel::Init(CMesh* mesh) 
{
	CSubModel* pSubModel = new CSubModel();

	V_RET_FOF(pSubModel->Init(mesh));

	m_vecSubModels.push_back(pSubModel);

	return true;
}

bool CModel::Init(std::string name) 
{
	CObjFileLoader* objFileLoader = new CObjFileLoader();
	std::vector<CMeshGeometry*> m_MeshGeometries;
	std::vector<CMeshMaterial*> m_MeshMaterials;

	std::string path("Resources\\");
	path = path.append(name);

	objFileLoader->ParseObjFile(path, m_MeshGeometries, m_MeshMaterials);
	
	if(m_MeshGeometries.size() < 1)
	{
		std::cout << "Number of meshes less than one. error" << std::endl;
		return false;
	}

	for(uint i = 0; i < m_MeshGeometries.size(); ++i)
	{
		CSubModel* pSubModel = new CSubModel();
		V_RET_FOF(pSubModel->Init(m_MeshGeometries[i]));
		m_vecSubModels.push_back(pSubModel);
	}
	
	m_WorldTransform = glm::scale(glm::vec3(1.f, 1.f, 1.f));

	return true;
}

void CModel::Release()
{
	std::vector<CSubModel*>::iterator it;
	for(it = m_vecSubModels.begin(); it < m_vecSubModels.end(); ++it)
	{
		(*it)->Release();
	}

	for(uint i = 0; i < m_MeshGeometries.size(); ++i)
	{
		SAFE_DELETE(m_MeshGeometries[i]);
	}
	m_MeshGeometries.clear();

	for(uint i = 0; i < m_MeshMaterials.size(); ++i)
	{
		SAFE_DELETE(m_MeshMaterials[i]);
	}
	m_MeshMaterials.clear();
}

void CModel::Draw(const glm::mat4& mView, const glm::mat4& mProj, COGLUniformBuffer* pUBTranform, COGLUniformBuffer* pUBMaterial) 
{
	TRANSFORM transform;
	transform.M = m_WorldTransform;
	transform.V = mView;
	transform.itM = glm::transpose(glm::inverse(m_WorldTransform));
	transform.MVP = mProj * mView * m_WorldTransform;

	pUBTranform->UpdateData(&transform);

	std::vector<CSubModel*>::iterator it;
	for(it = m_vecSubModels.begin(); it < m_vecSubModels.end(); ++it)
	{
		(*it)->Draw(pUBMaterial);
	}
}

void CModel::Draw(const glm::mat4& mView, const glm::mat4& mProj, COGLUniformBuffer* pUBTranform) 
{
	TRANSFORM transform;
	transform.M = m_WorldTransform;
	transform.V = mView;
	transform.itM = glm::transpose(glm::inverse(m_WorldTransform));
	transform.MVP = mProj * mView * m_WorldTransform;

	pUBTranform->UpdateData(&transform);

	std::vector<CSubModel*>::iterator it;
	for(it = m_vecSubModels.begin(); it < m_vecSubModels.end(); ++it)
	{
		(*it)->Draw();
	}
}

void CModel::SetMaterial(MATERIAL& mat)
{
	std::vector<CSubModel*>::iterator it;
	for(it = m_vecSubModels.begin(); it < m_vecSubModels.end(); ++it)
	{
		(*it)->SetMaterial(mat);
	}
}

MATERIAL& CModel::GetMaterial()
{
	std::vector<CSubModel*>::iterator it;
	for(it = m_vecSubModels.begin(); it < m_vecSubModels.end(); ++it)
	{
		return (*it)->GetMaterial();
	}

	MATERIAL* mat = new MATERIAL();
	return *mat;
}

glm::vec3 CModel::GetPositionWS()
{
	glm::vec4 temp = m_WorldTransform * glm::vec4(0.f, 0.f, 0.f, 1.f);
	temp = temp / temp.w;	
	return glm::vec3(temp);
}

