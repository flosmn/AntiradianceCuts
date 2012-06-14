#ifndef _C_POST_PROCESS_H_
#define _C_POST_PROCESS_H_

#include "GL/glew.h"

class COGLTexture2D;
class COGLSampler;
class COGLUniformBuffer;

class CProgram;
class CRenderTarget;

class CFullScreenQuad;

class CPostprocess
{
public:
	CPostprocess();
	~CPostprocess();

	bool Init();
	void Release();
	
	void Postprocess(COGLTexture2D* pTexture, CRenderTarget* result);

	void SetExposure(float exposure) { m_Exposure = exposure; UpdateUniformBuffer(); }
	void SetGamma(float gamma) { m_Gamma = gamma; UpdateUniformBuffer(); } 

private:
	void UpdateUniformBuffer();

	CProgram* m_pPostProcessProgram;
	CFullScreenQuad* m_pFullScreenQuad;

	COGLUniformBuffer* m_pUniformBuffer;
	COGLSampler* m_pPointSampler;
	
	float m_Gamma;
	float m_Exposure;
};

#endif // _C_POST_PROCESS_H_