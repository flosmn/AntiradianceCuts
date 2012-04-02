#ifndef _C_MESH_MATERIAL_H_
#define _C_MESH_MATERIAL_H_

#include "glm/glm.hpp"

#include "Structs.h"

#include <string>

class CMeshMaterial
{
public:
	CMeshMaterial();
	~CMeshMaterial();

	void SetMaterialName(std::string name) { m_Name = name; }
	void SetDiffuseColor(glm::vec4 color) { m_MaterialData.diffuseColor = color; }

	std::string GetMaterialName() { return m_Name; }
	glm::vec4 GetDiffuseColor() { return m_MaterialData.diffuseColor; }

	MATERIAL& GetMaterialData() { return m_MaterialData; }
	void SetMaterialData(MATERIAL& mat) { m_MaterialData = mat; }

	void PrintMaterial();

private:
	std::string m_Name;
	
	MATERIAL m_MaterialData;
};

#endif _C_MESH_MATERIAL_H_