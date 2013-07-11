#ifndef POSTPROCESS_H_
#define POSTPROCESS_H_

#include "GL/glew.h"

#include <memory>

#include "OGLResources/COGLTexture2D.h"
#include "OGLResources/COGLSampler.h"
#include "OGLResources/COGLProgram.h"

#include "CProgram.h"
#include "CRenderTarget.h"
#include "CRenderTargetLock.h"

#include "FullScreenQuad.h"
#include "CConfigManager.h"

class Postprocess
{
public:
	Postprocess(CConfigManager* configManager) :
		m_configManager(configManager)
	{
		m_fullScreenQuad.reset(new FullScreenQuad());
		m_program		.reset(new CProgram("Shaders\\PostProcess.vert", "Shaders\\PostProcess.frag"));
		m_pointSampler 	.reset(new COGLSampler(GL_NEAREST, GL_NEAREST, GL_REPEAT, GL_REPEAT, "CPostProcess.m_pPointSampler"));
		
		m_program->BindSampler(0, m_pointSampler.get());
	}
	
	void postprocess(COGLTexture2D* pTexture, CRenderTarget* target)
	{
		CRenderTargetLock lock(target);
		COGLBindLock lockProgram(m_program->GetGLProgram(), COGL_PROGRAM_SLOT);
		COGLBindLock lockTexture(pTexture, COGL_TEXTURE0_SLOT);

		{
			GLuint location = glGetUniformLocation(m_program->GetGLProgram()
				->GetResourceIdentifier(), "gamma");
			glUniform1f(location, m_configManager->GetConfVars()->Gamma);
		}
		{
			GLuint location = glGetUniformLocation(m_program->GetGLProgram()
				->GetResourceIdentifier(), "exposure");
			glUniform1f(location, m_configManager->GetConfVars()->Exposure);
		}

		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		m_fullScreenQuad->draw();
	}

private:
	std::unique_ptr<CProgram> m_program;
	std::unique_ptr<FullScreenQuad> m_fullScreenQuad;
	std::unique_ptr<COGLSampler> m_pointSampler;
	CConfigManager* m_configManager;
};

#endif // POSTPROCESS_H_