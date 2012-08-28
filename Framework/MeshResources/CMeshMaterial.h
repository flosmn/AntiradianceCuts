#ifndef _C_MESH_MATERIAL_H_
#define _C_MESH_MATERIAL_H_

#include "glm/glm.hpp"

#include "..\Structs.h"

#include <string>

class CMeshMaterial
{
public:
	CMeshMaterial();
	~CMeshMaterial();

	void SetMaterialName(const std::string& name) { m_Name = name; }
	void SetDiffuse(const glm::vec4& x) { m_MaterialData.diffuse = x; }
	void SetEmissive(const glm::vec4& x) { m_MaterialData.emissive = x; }

	std::string GetMaterialName() const { return m_Name; }
	glm::vec4 GetDiffuse() const { return m_MaterialData.diffuse; }
	glm::vec4 GetEmissive() const { return m_MaterialData.emissive; }

	MATERIAL GetMaterialData() const { return m_MaterialData; }
	void SetMaterialData(const MATERIAL& mat) { m_MaterialData = mat; }

	void PrintMaterial();

private:
	std::string m_Name;	
	MATERIAL m_MaterialData;
};

#endif _C_MESH_MATERIAL_H_