#include "Postprocessing.h"

#include "Macros.h"

#include "CUtils\ShaderUtil.h"

#include "CMeshResources\CFullScreenQuad.h"

Postprocessing::Postprocessing() 
{
	gamma = 2.2f;
	exposure = 1.0f;

	m_pFullScreenQuad = new CFullScreenQuad();
}

Postprocessing::~Postprocessing() 
{
	SAFE_DELETE(m_pFullScreenQuad);
}

bool Postprocessing::Init()
{
	V_RET_FOF(m_pFullScreenQuad->Init());	

	glGenSamplers(1, &m_Sampler);
	 
	postProcessingProgram = CreateProgram("Shaders\\PostProcessing.vert", "Shaders\\PostProcessing.frag");

	uniformNumberOfLightPaths = glGetUniformLocation(postProcessingProgram, "numberOfLightPaths");
	uniformExposure = glGetUniformLocation(postProcessingProgram, "exposure");
	uniformGamma = glGetUniformLocation(postProcessingProgram, "gamma");
	uniformSampler = glGetUniformLocation(postProcessingProgram, "sampler");
	uniformWidth = glGetUniformLocation(postProcessingProgram, "width");
	uniformHeight = glGetUniformLocation(postProcessingProgram, "height");

	return true;
}

void Postprocessing::Release()
{
	m_pFullScreenQuad->Release();

	glDeleteSamplers(1, &m_Sampler);
}

void Postprocessing::Postprocess(GLuint textureSource, GLuint width, GLuint height, int numberOfLightPaths)
{
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);
	glPolygonOffset(0.0f, 0.0f);
	glViewport(0, 0, width, height);

	glUseProgram(postProcessingProgram);
		
	glUniform1f(uniformWidth, (GLfloat)width);
	glUniform1f(uniformHeight, (GLfloat)height);
	glUniform1f(uniformNumberOfLightPaths, (GLfloat)numberOfLightPaths);	
	glUniform1f(uniformGamma, gamma);
	glUniform1f(uniformExposure, exposure);
	
	glUniform1i(uniformSampler, 0);
	
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, textureSource);
	glBindSampler(textureSource, m_Sampler);
	
	m_pFullScreenQuad->Draw();
	
	glActiveTexture(GL_TEXTURE0);
	glBindSampler(m_Sampler, 0);
	glBindTexture(GL_TEXTURE_2D, 0);
	
	glUseProgram(0);
}
