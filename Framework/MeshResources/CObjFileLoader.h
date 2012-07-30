#ifndef _C_OBJ_FILE_LOADER_H_
#define _C_OBJ_FILE_LOADER_H_

#include "glm/glm.hpp"

#include <vector>

class CMeshGeometry;
class CMeshMaterial;
class CMeshTriangleFace;

class CTimer;

class CTempMesh
{
public:
	CTempMesh() { }
	~CTempMesh() { }

	std::vector<CMeshTriangleFace> m_Triangles;
	CMeshMaterial* m_Material;
};

class CObjFileLoader
{
public:
	CObjFileLoader();
	~CObjFileLoader();

	void ParseObjFile(std::string file, std::vector<CMeshGeometry*>& meshes,
		std::vector<CMeshMaterial*>& materials);

private:
	bool ParseObjFileLine(const std::string& line, const std::vector<CMeshMaterial*>& materials);
	bool ParseMtlFileLine(const std::string& line, std::vector<CMeshMaterial*>& materials);

	void CreateMeshGeometry(CTempMesh* pTempMesh, CMeshGeometry* pMeshGeometry);

	bool m_bNewSubMesh;
	CTempMesh* m_pCurrentTempMesh;
	std::vector<CTempMesh*> m_TempMeshes;
	CMeshMaterial* m_pCurrentMaterial;

	std::vector<glm::vec4> m_Positions;
	std::vector<glm::vec3> m_Normals;
	std::vector<glm::vec3> m_TexCoords;

	CTimer* timer;
	double parseVertexInfoTime;
	double parseFaceInfoTime;
	double parseMaterialInfoTime;
};

#endif _C_OBJ_FILE_LOADER_H_
