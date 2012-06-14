#ifndef _C_OCL_TEXTURE_2D_H
#define _C_OCL_TEXTURE_2D_H

#include "COCLResource.h"

#include "CL/cl.h"

class COCLContext;
class COGLTexture2D;

class COCLTexture2D : public COCLResource
{
public:
	COCLTexture2D(COCLContext* pContext, const std::string& debugName);
	~COCLTexture2D();

	virtual bool Init(COGLTexture2D* pGLTexture);
	virtual void Release();

	cl_mem* GetCLTexture() { CheckInitialized("COCLTexture2D.GetCLTexture()"); return &m_Texture; }

	void Lock();
	void Unlock();

private:
	cl_mem m_Texture;

	COGLTexture2D* m_pGLTexture;
	COCLContext* m_pContext;
};

#endif // _C_OCL_TEXTURE_2D_H