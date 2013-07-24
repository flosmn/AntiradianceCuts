#ifndef _C_MATERIAL_BUFFER_H_
#define _C_MATERIAL_BUFFER_H_

#include "Material.h"

#include <unordered_map>
#include <string>
#include <vector>
#include <memory>

class COGLTextureBuffer;

class CMaterialBuffer
{
public:
	CMaterialBuffer();
	~CMaterialBuffer();
	
	bool FillOGLMaterialBuffer();
	
	COGLTextureBuffer* GetOGLMaterialBuffer() { return m_oglMaterialBuffer.get(); }
	
	// add the material mat to the material buffer and returns the index of the material in the buffer
	int AddMaterial(std::string const& name, MATERIAL const& mat);

	int GetIndexOfMaterial(std::string const& name);

	// returns the material with the index i
	MATERIAL* GetMaterial(int i);

	std::vector<MATERIAL> const& getMaterials() { return m_Materials; }
	
private:
	std::unordered_map<std::string, int> m_MapMatNameToIndex;
	std::vector<MATERIAL> m_Materials;

	std::unique_ptr<COGLTextureBuffer> m_oglMaterialBuffer;
};

#endif _C_MATERIAL_BUFFER_H_
