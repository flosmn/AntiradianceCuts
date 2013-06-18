#ifndef _C_POST_PROCESS_H_
#define _C_POST_PROCESS_H_

#include "GL/glew.h"

#include <memory>

class COGLTexture2D;
class COGLSampler;
class COGLUniformBuffer;

class CProgram;
class CRenderTarget;

class CFullScreenQuad;
class CConfigManager;

class CPostprocess
{
public:
	CPostprocess(CConfigManager* pConfigManager);
	~CPostprocess();

	void Postprocess(COGLTexture2D* pTexture, CRenderTarget* result);

	void SetExposure(float exposure) { m_Exposure = exposure; UpdateUniformBuffer(); }
	void SetGamma(float gamma) { m_Gamma = gamma; UpdateUniformBuffer(); } 

private:
	void UpdateUniformBuffer();

	std::unique_ptr<CProgram> m_program;
	std::unique_ptr<CFullScreenQuad> m_fullScreenQuad;
	std::unique_ptr<COGLUniformBuffer> m_uniformBuffer;
	std::unique_ptr<COGLSampler> m_pointSampler;
	
	float m_Gamma;
	float m_Exposure;

	CConfigManager* m_pConfigManager;
};

#endif // _C_POST_PROCESS_H_