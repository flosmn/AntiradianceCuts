#ifndef _C_PROGRAM_H_
#define _C_PROGRAM_H_

typedef unsigned int uint;

#include <string>
#include <vector>
#include <memory>

class COGLProgram;
class COGLUniformBuffer;
class COGLSampler;

class CProgram
{
public:
	CProgram(std::string const& VS, std::string const& FS, std::string const& debugName = "");
	CProgram(std::string const& VS, std::string const& FS, std::vector<std::string> headerFiles, std::string const& debugName = "");
	~CProgram();
	
	void BindUniformBuffer(COGLUniformBuffer* pGLUniformBuffer, std::string strUniformBlockNam);
	void BindSampler(uint samplerSlot, COGLSampler* pGLSampler);

	COGLProgram* GetGLProgram();

private:
	std::unique_ptr<COGLProgram> m_program;
};

#endif // _C_PROGRAM_H_