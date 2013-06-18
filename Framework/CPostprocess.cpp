#include "CPostprocess.h"

#include "Macros.h"

#include "CProgram.h"
#include "CRenderTarget.h"
#include "CRenderTargetLock.h"

#include "CConfigManager.h"

#include "Utils\ShaderUtil.h"

#include "OGLResources\COGLProgram.h"
#include "OGLResources\COGLTexture2D.h"
#include "OGLResources\COGLSampler.h"
#include "OGLResources\COGLUniformBuffer.h"

#include "MeshResources\CFullScreenQuad.h"

struct POST_PROCESS
{
	float one_over_gamma;
	float exposure;
};

CPostprocess::CPostprocess(CConfigManager* pConfigManager) 
{
	m_Gamma = 2.2f;
	m_Exposure = 1.0f;

	POST_PROCESS pp;
	pp.one_over_gamma = 1.f / m_Gamma;
	pp.exposure = m_Exposure;
	
	m_fullScreenQuad.reset(new CFullScreenQuad());
	m_program		.reset(new CProgram("Shaders\\PostProcess.vert", "Shaders\\PostProcess.frag"));
	m_pointSampler 	.reset(new COGLSampler(GL_NEAREST, GL_NEAREST, GL_REPEAT, GL_REPEAT, "CPostProcess.m_pPointSampler"));
	m_uniformBuffer .reset(new COGLUniformBuffer(sizeof(POST_PROCESS), &pp, GL_STATIC_DRAW, "CPostProcess.m_pUniformBuffer"));
	
	m_program->BindSampler(0, m_pointSampler.get());
	m_program->BindUniformBuffer(m_uniformBuffer.get(), "postprocess");

	m_pConfigManager = pConfigManager;
}

CPostprocess::~CPostprocess() 
{
}

void CPostprocess::Postprocess(COGLTexture2D* pTexture, CRenderTarget* pTarget)
{
	UpdateUniformBuffer();
	
	CRenderTargetLock lock(pTarget);

	COGLBindLock lockProgram(m_program->GetGLProgram(), COGL_PROGRAM_SLOT);

	COGLBindLock lockTexture(pTexture, COGL_TEXTURE0_SLOT);
	
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	m_fullScreenQuad->Draw();
}

void CPostprocess::UpdateUniformBuffer()
{
	float gamma = 1.f;
	if(m_pConfigManager->GetConfVars()->UseGammaCorrection)
		gamma = m_pConfigManager->GetConfVars()->Gamma;

	float exposure = 1.f;
	if(m_pConfigManager->GetConfVars()->UseToneMapping)
		exposure = m_pConfigManager->GetConfVars()->Exposure;
	
	POST_PROCESS pp;
	pp.one_over_gamma = 1.f / gamma;
	pp.exposure = exposure;
	m_uniformBuffer->UpdateData(&pp);
}
