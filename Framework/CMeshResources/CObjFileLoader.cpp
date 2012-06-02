#include "CObjFileLoader.h"

#include "CMeshGeometry.h"
#include "CMeshMaterial.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <set>
#include <map>


CObjFileLoader::CObjFileLoader()
{
	m_pCurrentTempMesh = nullptr;
}

CObjFileLoader::~CObjFileLoader()
{

}

void CObjFileLoader::ParseObjFile(std::string name, std::vector<CMeshGeometry*>& meshes,
		std::vector<CMeshMaterial*>& materials)
{
	char line[1024];

	std::string mtlFileName = name;
	mtlFileName.append(".mtl");
	
	// parse .mtl file
	std::ifstream mtlFile(mtlFileName.c_str(), std::ifstream::in);
	if(!mtlFile){
		std::cout << "Could not open mtl file " << mtlFileName.c_str() << std::endl;
		return;
	}
	while(mtlFile.good())
	{
		mtlFile.getline(line,1023);
		if(!ParseMtlFileLine(std::string(line), materials))
		{
			std::cout << "Error occured while parsing file " << mtlFileName.c_str() << std::endl;
		}
	}
	mtlFile.close();

	// parse .obj file
	std::string objFileName = name;
	objFileName.append(".obj");
		
	std::ifstream objFile(objFileName.c_str(), std::ifstream::in);
	if(!objFile){
		std::cout << "Could not open given file " << objFileName.c_str() << std::endl;
		return;
	}
	while(objFile.good())
	{
		objFile.getline(line,1023);
		if(!ParseObjFileLine(std::string(line), materials))
		{
			std::cout << "Error occured while parsing file " << objFileName.c_str() << std::endl;
		}
	}
	objFile.close();

	// create CMeshGeometry out of TempMeshes
	for(uint i = 0; i < m_TempMeshes.size(); ++i)
	{
		CMeshGeometry* meshGeometry = new CMeshGeometry();
		CreateMeshGeometry(m_TempMeshes[i], meshGeometry);
		meshes.push_back(meshGeometry);
	}
}

void CObjFileLoader::CreateMeshGeometry(CTempMesh* pTempMesh, CMeshGeometry* pMeshGeometry)
{
	std::set<CMeshVertex*, CMeshVertex_Compare> setVertices;
	std::set<CMeshVertex*, CMeshVertex_Compare>::iterator set_it;

	std::vector<CMeshTriangleFace*>::iterator face_it;
	for(face_it = pTempMesh->m_Triangles.begin(); face_it < pTempMesh->m_Triangles.end(); ++face_it)
	{
		CMeshTriangleFace* face = *face_it;
		setVertices.insert(face->vertex0);
		setVertices.insert(face->vertex1);
		setVertices.insert(face->vertex2);
	}

	uint nGLVertices = setVertices.size();
	glm::vec4* pGLPositions = nullptr;
	glm::vec3* pGLNormals = nullptr;
	glm::vec3* pGLTexCoords = nullptr;
	
	std::map<CMeshVertex, uint, CMeshVertex_Compare> mapMeshVertexIndex;

	if(m_Positions.size() > 0)
		pGLPositions = new glm::vec4[nGLVertices];
	if(m_Normals.size() > 0)
		pGLNormals = new glm::vec3[nGLVertices];
	if(m_TexCoords.size() > 0)
		pGLTexCoords = new glm::vec3[nGLVertices];

	uint index = 0;
	for(set_it = setVertices.begin(); index < nGLVertices; ++set_it)
	{
		CMeshVertex vertex = **set_it;
		mapMeshVertexIndex[vertex] = index + 1;

		if(m_Positions.size() > 0 && vertex.positionDataIndex != 0)
		{
			if(vertex.positionDataIndex > (int)m_Positions.size())
			{
				std::cout << "position data index out of range" << std::endl;
				return;
			}
			pGLPositions[index]	= m_Positions[vertex.positionDataIndex - 1];
		}
		if(m_Normals.size() > 0 && vertex.normalDataIndex != 0)
		{
			if(vertex.normalDataIndex > (int)m_Normals.size())
			{
				std::cout << "normal data index out of range" << std::endl;
				return;
			}
			pGLNormals[index] = m_Normals[vertex.normalDataIndex - 1];
		}
		if(m_TexCoords.size() > 0 && vertex.texCoordDataIndex != 0)
		{
			if(vertex.texCoordDataIndex > (int)m_TexCoords.size())
			{
				std::cout << "texCoord data index out of range" << std::endl;
				return;
			}
			pGLTexCoords[index]	= m_TexCoords[vertex.texCoordDataIndex - 1];
		}

		index++;
	}

	ushort* pGLIndexData = new ushort[3 * pTempMesh->m_Triangles.size()];
	int face_index = 0;
	for(face_it = pTempMesh->m_Triangles.begin(); face_it < pTempMesh->m_Triangles.end(); ++face_it)
	{
		CMeshTriangleFace* face = *face_it;

		// obj. file orientation is counter GL orientation
		pGLIndexData[3 * face_index + 0] = mapMeshVertexIndex[*(face->vertex2)] - 1;
		pGLIndexData[3 * face_index + 1] = mapMeshVertexIndex[*(face->vertex1)] - 1;
		pGLIndexData[3 * face_index + 2] = mapMeshVertexIndex[*(face->vertex0)] - 1;
		face_index++;
	}

	pMeshGeometry->SetNumberOfVertices(nGLVertices);
	pMeshGeometry->SetNumberOfFaces(pTempMesh->m_Triangles.size());
	pMeshGeometry->SetMeshMaterial(pTempMesh->m_Material);
	pMeshGeometry->SetIndexData(pGLIndexData);
	
	if(pGLPositions)
		pMeshGeometry->SetPositionData(pGLPositions);
	
	if(pGLNormals) {
		pMeshGeometry->SetNormalData(pGLNormals);
	}
	else {
		// create flat normals
		pGLNormals = new glm::vec3[nGLVertices];
		for(uint face_index = 0; face_index < pTempMesh->m_Triangles.size(); face_index++)
		{
			ushort index0 = pGLIndexData[3 * face_index + 0];
			ushort index1 = pGLIndexData[3 * face_index + 1];
			ushort index2 = pGLIndexData[3 * face_index + 2];

			glm::vec4 vertex0 = pGLPositions[index0];
			glm::vec4 vertex1 = pGLPositions[index1];
			glm::vec4 vertex2 = pGLPositions[index2];
			
			glm::vec3 u = glm::normalize(glm::vec3(vertex1 - vertex0));
			glm::vec3 v = glm::normalize(glm::vec3(vertex2 - vertex0));

			glm::vec3 normal = -glm::normalize(glm::cross(u, v));

			pGLNormals[index0] = normal;
			pGLNormals[index1] = normal;
			pGLNormals[index2] = normal;
		}

		pMeshGeometry->SetNormalData(pGLNormals);
	}
	
	if(pGLTexCoords)
		pMeshGeometry->SetTexCoordData(pGLTexCoords);	
}

bool CObjFileLoader::ParseObjFileLine(std::string line, const std::vector<CMeshMaterial*>& materials)
{
	std::string op;
	if (line.empty())
		return true;

	std::stringstream ss(std::stringstream::in  | std::stringstream::out);
	ss.str(line);

	ss >> op;

	if(op.size() == 0 || op == "") return true;

	if (op[0] == '#')
	{
		return true;
	}
	else if (op.compare("v") == 0) 
	{
		float x = 0.f, y = 0.f, z = 0.f, w = 1.f;
		ss >> x >> y >> z >> w;

		m_Positions.push_back(glm::vec4(x, y, z, w));
	}
	else if (op.compare("vn") == 0) 
	{
		m_bNewSubMesh = true;

		float x = 0.f, y = 0.f, z = 0.f;
		ss >> x >> y >> z;

		m_Normals.push_back(glm::vec3(x, y, z));
	}
	else if (op.compare("vt") == 0)
	{
		m_bNewSubMesh = true;

		float x = 0.f, y = 0.f, z = 0.f;
		ss >> x >> y >> z;

		m_TexCoords.push_back(glm::vec3(x, y, z));
	}
	else if (op.compare("f") == 0)
	{
		if(m_pCurrentTempMesh == 0)
		{
			m_pCurrentTempMesh = new CTempMesh();
			m_TempMeshes.push_back(m_pCurrentTempMesh);
		}
		
		std::string indexData[4];
		for(int i = 0; i < 4; ++i) indexData[i] = "";

		ss >> indexData[0] >> indexData[1] >> indexData[2] >> indexData[3];
				
		if(indexData[3] != "") 
		{
			// two triangles
			std::cout << "quad faces are not implemented yet" << std::endl;
		}
		else
		{
			// one triangle
			CMeshTriangleFace* triangleFace = new CMeshTriangleFace();
			CMeshVertex** vertices = new CMeshVertex*[3];

			// for each vertex
			for(int i = 0; i < 3; ++i)
			{
				vertices[i] = new CMeshVertex();
								
				std::istringstream issToken(indexData[i]);
				std::string token;

				// for each vertex data
				for(int j = 0; j < 3; ++j)
				{
					if(!std::getline( issToken, token, '/' )) break;
					std::stringstream ss(std::stringstream::in  | std::stringstream::out);
					ss.str(token);
					if(j==0) ss >> vertices[i]->positionDataIndex;
					if(j==1) ss >> vertices[i]->texCoordDataIndex;
					if(j==2) ss >> vertices[i]->normalDataIndex;
				}
			}
			
			triangleFace->vertex0 = vertices[0];
			triangleFace->vertex1 = vertices[1];
			triangleFace->vertex2 = vertices[2];
			
			m_pCurrentTempMesh->m_Triangles.push_back(triangleFace);
		}
	}
	else if (op.compare("usemtl") == 0)
	{
		m_pCurrentTempMesh = new CTempMesh();
		m_TempMeshes.push_back(m_pCurrentTempMesh);
		
		std::string materialName;
		ss >> materialName;

		for(uint i = 0; i < materials.size(); ++i)
		{
			if(materials[i]->GetMaterialName() == materialName)
			{
				m_pCurrentTempMesh->m_Material = materials[i];
				return true;
			}
		}

		std::cout << "material not found: " << materialName << std::endl;
	}
	
	return true;
}

bool CObjFileLoader::ParseMtlFileLine(std::string line, std::vector<CMeshMaterial*>& materials)
{
	std::string op;
	if (line.empty())
		return true;

	std::stringstream ss(std::stringstream::in  | std::stringstream::out);
	ss.str(line);

	ss >> op;

	if(op.size() == 0 || op == "") return true;

	if (op[0] == '#')
	{
		return true;
	}
	else if (op.compare("newmtl") == 0) 
	{
		m_pCurrentMaterial = new CMeshMaterial();
		materials.push_back(m_pCurrentMaterial);
		
		std::string materialName;
		ss >> materialName;

		m_pCurrentMaterial->SetMaterialName(materialName);
	}
	else if (op.compare("Kd") == 0) 
	{
		float r = 0.f, g = 0.f, b = 0.f, a = 0.f;
		ss >> r >> g >> b >> a;

		m_pCurrentMaterial->SetDiffuseColor(glm::vec4(r, g, b, a));
	}
	
	return true;
}