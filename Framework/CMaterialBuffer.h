#ifndef _C_MATERIAL_BUFFER_H_
#define _C_MATERIAL_BUFFER_H_

#include "Material.h"

#include <unordered_map>
#include <string>
#include <vector>

class COGLTextureBuffer;

class CMaterialBuffer
{
public:
	CMaterialBuffer();
	~CMaterialBuffer();

	bool InitOGLMaterialBuffer();
	void ReleaseOGLMaterialBuffer();

	COGLTextureBuffer* GetOGLMaterialBuffer();

	// add the material mat to the material buffer and returns the index of the material in the buffer
	int AddMaterial(const std::string& name, const MATERIAL& mat);

	int GetIndexOfMaterial(const std::string& name);

	// returns the material with the index i
	MATERIAL* GetMaterial(int i);
	
private:
	std::unordered_map<std::string, int> m_MapMatNameToIndex;
	std::vector<MATERIAL> m_Materials;

	COGLTextureBuffer* m_pOGLMaterialBuffer;
};

#endif _C_MATERIAL_BUFFER_H_