#include "CMeshMaterial.h"

#include "..\Utils\Util.h"

#include <iostream>

CMeshMaterial::CMeshMaterial()
{
	m_Name = std::string("no name");
}

CMeshMaterial::~CMeshMaterial()
{
	
}

void CMeshMaterial::PrintMaterial()
{
	std::cout << "Material Name: " << m_Name << std::endl;
	std::cout << "Diffuse: " << AsString(m_MaterialData.diffuse) << std::endl;
	std::cout << "Emissive: " << AsString(m_MaterialData.emissive) << std::endl;
}