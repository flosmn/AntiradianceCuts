#include "CModel.h"

#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>

#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include <assimp/importer.hpp>

#include "..\Macros.h"
#include "..\Structs.h"

#include "CMesh.h"
#include "CSubModel.h"
#include "CMeshGeometry.h"
#include "CMeshMaterial.h"
#include "CObjFileLoader.h"
#include "..\CTriangle.h"

#include "..\CTimer.h"

#include "..\Utils\GLErrorUtil.h"

#include "..\OGLResources\COGLVertexBuffer.h"
#include "..\OGLResources\COGLVertexArray.h"
#include "..\OGLResources\COGLBindLock.h"
#include "..\OGLResources\COGLUniformBuffer.h"

#include <fstream>


CModel::CModel()
	: m_Mesh(nullptr)
{
}

CModel::~CModel()
{
	for(int i = 0; i < m_SubModels.size(); ++i)
		delete m_SubModels[i];

	m_SubModels.clear();
};

bool CModel::Init(CMesh* mesh) 
{
	CSubModel* pSubModel = new CSubModel();

	V_RET_FOF(pSubModel->Init(mesh));

	m_SubModels.push_back(pSubModel);

	return true;
}

bool CModel::Init(std::string name) 
{
	std::cout << "Start loading model " << name << std::endl;
	CTimer timer(CTimer::CPU);
	
	CObjFileLoader* objFileLoader = new CObjFileLoader();
	std::vector<CMeshGeometry*> m_MeshGeometries;
	std::vector<CMeshMaterial*> m_MeshMaterials;
	
	timer.Start();
	
	// Create an instance of the Importer class
	Assimp::Importer importer;
	const aiScene* scene = NULL;
	
	//check if file exists
	std::string path("Resources\\");
	path = path.append(name);
	path = path.append(".obj");
	std::ifstream fin(path.c_str());
    if(!fin.fail()) {
        fin.close();
    }
    else{
        printf("Couldn't open file: %s\n", path);
        printf("%s\n", importer.GetErrorString());
        return false;
    }
 
	std::string pFile(path);
	scene = importer.ReadFile( pFile, aiProcess_Triangulate | aiProcess_GenNormals | aiProcess_FlipWindingOrder);
 
    // If the import failed, report it
    if( !scene)
    {
        printf("%s\n", importer.GetErrorString());
        return false;
    }
 
    // Now we can access the file's contents.
    timer.Stop("Import with assimp");
	
	timer.Start();
    // For each mesh
    for (unsigned int n = 0; n < scene->mNumMeshes; ++n)
    {
		CMeshGeometry meshGeometry;
		CMeshMaterial material;

        const struct aiMesh* mesh = scene->mMeshes[n];
 
        // create array with faces
        // have to convert from Assimp format to array
        uint *faceArray = new uint[mesh->mNumFaces * 3];
        uint faceIndex = 0;
 
        for (unsigned int t = 0; t < mesh->mNumFaces; ++t) {
            const struct aiFace* face = &mesh->mFaces[t];
			faceArray[3 * faceIndex + 0] = face->mIndices[0];
			faceArray[3 * faceIndex + 1] = face->mIndices[1];
			faceArray[3 * faceIndex + 2] = face->mIndices[2];
            faceIndex++;
        }
		meshGeometry.SetNumberOfFaces(scene->mMeshes[n]->mNumFaces);
		meshGeometry.SetNumberOfVertices(mesh->mNumVertices);
		meshGeometry.SetIndexData(faceArray);

        // buffer for vertex positions
		glm::vec4* positions = new glm::vec4[mesh->mNumVertices];
        if (mesh->HasPositions()) {			
			for(uint i = 0; i < mesh->mNumVertices; ++i)
				positions[i] = glm::vec4(mesh->mVertices[i].x, mesh->mVertices[i].y, mesh->mVertices[i].z, 1.f);

			meshGeometry.SetPositionData(positions);
        }
 		
        // buffer for vertex normals
		glm::vec3* normals = new glm::vec3[mesh->mNumVertices];
        if (mesh->HasNormals()) {
			for(uint i = 0; i < mesh->mNumVertices; ++i)
				normals[i] = glm::vec3(mesh->mNormals[i].x, mesh->mNormals[i].y, mesh->mNormals[i].z);

			meshGeometry.SetNormalData(normals);
        }
 		
        // create material uniform buffer
        struct aiMaterial *mtl = scene->mMaterials[mesh->mMaterialIndex];
         
        glm::vec4 diffuse = glm::vec4(0.8f, 0.8f, 0.8f, 1.f);
        aiColor4D d;
        if(AI_SUCCESS == aiGetMaterialColor(mtl, AI_MATKEY_COLOR_DIFFUSE, &d))
            diffuse = glm::vec4(d.r, d.g, d.b, d.a);
        
		material.SetDiffuseColor(diffuse);
		meshGeometry.SetMeshMaterial(material);

		CSubModel* subModel = new CSubModel();
		subModel->Init(meshGeometry);

        m_SubModels.push_back(subModel);

		// face, position and normal data will be deleted by CMeshGeometry
    }
	timer.Stop("Convert geometry info");

	SetWorldTransform(glm::scale(glm::vec3(1.f, 1.f, 1.f)));
	
	return true;
}

void CModel::Release()
{
	for(int i = 0; i < m_SubModels.size(); ++i)
		m_SubModels[i]->Release();
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
	for(it = m_SubModels.begin(); it < m_SubModels.end(); ++it)
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
	for(it = m_SubModels.begin(); it < m_SubModels.end(); ++it)
	{
		(*it)->Draw();
	}
}

glm::vec3 CModel::GetPositionWS()
{
	glm::vec4 temp = m_WorldTransform * glm::vec4(0.f, 0.f, 0.f, 1.f);
	temp = temp / temp.w;	
	return glm::vec3(temp);
}

void CModel::SetMaterial(const MATERIAL& mat)
{
	std::vector<CSubModel*>::iterator it;
	for(it = m_SubModels.begin(); it < m_SubModels.end(); ++it)
	{
		(*it)->SetMaterial(mat);
	}
}

MATERIAL CModel::GetMaterial() const
{
	for(uint i = 0; i < m_SubModels.size(); ++i)
	{
		return m_SubModels[i]->GetMaterial();
	}

	MATERIAL mat;
	return mat;
}

void CModel::SetWorldTransform(const glm::mat4& transform)
{
	m_WorldTransform = transform;

	for(uint i = 0; i < m_SubModels.size(); ++i)
	{
		m_SubModels[i]->SetWorldTransform(m_WorldTransform);
	}
}