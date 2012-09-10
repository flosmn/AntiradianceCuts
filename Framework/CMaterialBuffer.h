#ifndef _C_MATERIAL_BUFFER_H_
#define _C_MATERIAL_BUFFER_H_

#include "Material.h"

#include <unordered_map>
#include <string>
#include <vector>

class COGLTextureBuffer;

class COCLContext;
class COCLBuffer;

class CMaterialBuffer
{
public:
	CMaterialBuffer(COCLContext* pContext);
	~CMaterialBuffer();

	bool InitOGLMaterialBuffer();
	void ReleaseOGLMaterialBuffer();

	bool InitOCLMaterialBuffer();
	void ReleaseOCLMaterialBuffer();

	COGLTextureBuffer* GetOGLMaterialBuffer();
	COCLBuffer* GetOCLMaterialBuffer();
	
	// add the material mat to the material buffer and returns the index of the material in the buffer
	int AddMaterial(const std::string& name, const MATERIAL& mat);

	int GetIndexOfMaterial(const std::string& name);

	// returns the material with the index i
	MATERIAL* GetMaterial(int i);
	
private:
	std::unordered_map<std::string, int> m_MapMatNameToIndex;
	std::vector<MATERIAL> m_Materials;

	COGLTextureBuffer* m_pOGLMaterialBuffer;
	COCLBuffer* m_pOCLMaterialBuffer;
};

#endif _C_MATERIAL_BUFFER_H_