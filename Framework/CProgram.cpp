#include "CProgram.h"

#include "Macros.h"

#include "OGLResources\COGLProgram.h"
#include "OGLResources\COGLUniformBuffer.h"
#include "OGLResources\COGLSampler.h"

CProgram::CProgram(std::string const& VS, std::string const& FS, std::string const& debugName)
{
	std::vector<std::string> headerFiles;
	m_program.reset(new COGLProgram(VS, "", FS, headerFiles, debugName));
}

CProgram::CProgram(std::string const& VS, std::string const& FS, std::vector<std::string> headerFiles, std::string const& debugName)
{
	m_program.reset(new COGLProgram(VS, "", FS, headerFiles, debugName));
}

CProgram::~CProgram()
{
}

void CProgram::BindUniformBuffer(COGLUniformBuffer* pGLUniformBuffer, std::string strUniformBlockName)
{
	m_program->BindUniformBuffer(pGLUniformBuffer, strUniformBlockName);
}

void CProgram::BindSampler(uint samplerSlot, COGLSampler* pSampler)
{
	m_program->BindSampler(samplerSlot, pSampler);
}

COGLProgram* CProgram::GetGLProgram()
{
	return m_program.get();
}
