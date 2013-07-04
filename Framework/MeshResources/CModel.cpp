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
#include "CObjFileLoader.h"

#include "..\Triangle.h"
#include "..\CMaterialBuffer.h"

#include "..\CTimer.h"

#include "..\Utils\GLErrorUtil.h"

#include "..\OGLResources\COGLVertexBuffer.h"
#include "..\OGLResources\COGLVertexArray.h"
#include "..\OGLResources\COGLBindLock.h"
#include "..\OGLResources\COGLUniformBuffer.h"

#include <fstream>
#include <sstream>


CModel::CModel(CMesh* mesh)
{
	m_subModels.emplace_back(new CSubModel(mesh));
}

CModel::CModel(std::string name, std::string ext, CMaterialBuffer* pMaterialBuffer)
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
	std::stringstream ss;
	ss << "Resources\\" << name <<"." << ext;

	std::ifstream fin(ss.str().c_str());
    if(!fin.fail()) {
        fin.close();
    }
    else{
		printf("Couldn't open file: %s\n", ss.str().c_str());
        printf("%s\n", importer.GetErrorString());
        return;
    }
 
	std::string pFile(ss.str());
	scene = importer.ReadFile( pFile, aiProcess_Triangulate | aiProcess_GenNormals  | aiProcess_FlipWindingOrder);
 
    // If the import failed, report it
    if( !scene)
    {
        printf("%s\n", importer.GetErrorString());
        return;
    }
 
    // Now we can access the file's contents.
    timer.Stop("Import with assimp");
	
	timer.Start();
    // For each mesh
    for (unsigned int n = 0; n < scene->mNumMeshes; ++n)
    {
		CMeshGeometry meshGeometry;
		
        const struct aiMesh* mesh = scene->mMeshes[n];
 
		 // create material uniform buffer
        struct aiMaterial *mtl = scene->mMaterials[mesh->mMaterialIndex];
        
		aiColor4D d;
		aiString matName;
		float e = 0.f;

		std::string name = "noname";
		if(AI_SUCCESS == mtl->Get(AI_MATKEY_NAME, matName))
			name = std::string(matName.C_Str());

        glm::vec4 diffuse = glm::vec4(0.f);
		if(AI_SUCCESS == mtl->Get(AI_MATKEY_COLOR_DIFFUSE, d))
            diffuse = glm::vec4(d.r, d.g, d.b, d.a);

		glm::vec4 specular = glm::vec4(0.f);
		if(AI_SUCCESS == mtl->Get(AI_MATKEY_COLOR_SPECULAR, d))
			specular = glm::vec4(d.r, d.g, d.b, d.a);

		float exponent = 0.f;
		if(AI_SUCCESS == mtl->Get(AI_MATKEY_SHININESS, e))
			exponent = e;
		
		glm::vec4 emissive = glm::vec4(0.f);
		if(AI_SUCCESS == mtl->Get(AI_MATKEY_COLOR_EMISSIVE, d))
            emissive = glm::vec4(d.r, d.g, d.b, d.a);
        
		MATERIAL mat;
		mat.emissive = emissive;
		mat.diffuse = diffuse;
		mat.specular = specular;
		mat.exponent = exponent;

		int materialIndex = pMaterialBuffer->AddMaterial(name, mat);
		meshGeometry.SetMaterialIndex(materialIndex);

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
				positions[i] = glm::vec4(mesh->mVertices[i].x, mesh->mVertices[i].y, mesh->mVertices[i].z, float(materialIndex));

			meshGeometry.SetPositionData(positions);
        }
 		
        // buffer for vertex normals
		glm::vec3* normals = new glm::vec3[mesh->mNumVertices];
        if (mesh->HasNormals()) {
			for(uint i = 0; i < mesh->mNumVertices; ++i)
				normals[i] = glm::vec3(mesh->mNormals[i].x, mesh->mNormals[i].y, mesh->mNormals[i].z);

			meshGeometry.SetNormalData(normals);
        }
 		
		m_subModels.emplace_back(new CSubModel(meshGeometry, pMaterialBuffer));

		// face, position and normal data will be deleted by CMeshGeometry
    }
	timer.Stop("Convert geometry info");

	SetWorldTransform(glm::scale(glm::vec3(1.f, 1.f, 1.f)));
}

CModel::~CModel()
{
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
	for(it = m_subModels.begin(); it < m_subModels.end(); ++it)
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
	for(it = m_subModels.begin(); it < m_subModels.end(); ++it)
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

void CModel::SetWorldTransform(const glm::mat4& transform)
{
	m_WorldTransform = transform;

	for(uint i = 0; i < m_subModels.size(); ++i)
	{
		m_subModels[i]->SetWorldTransform(m_WorldTransform);
	}
}
