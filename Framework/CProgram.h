#ifndef _C_PROGRAM_H_
#define _C_PROGRAM_H_

typedef unsigned int uint;

#include <string>
#include <vector>

class COGLProgram;
class COGLUniformBuffer;
class COGLSampler;

class CProgram
{
public:
	CProgram(const std::string debugName, const std::string VS, const std::string FS);
	CProgram(const std::string debugName, const std::string VS, const std::string FS, std::vector<std::string> headerFiles);
	~CProgram();

	bool Init();
	void Release();

	void BindUniformBuffer(COGLUniformBuffer* pGLUniformBuffer, std::string strUniformBlockNam);
	void BindSampler(uint samplerSlot, COGLSampler* pGLSampler);

	COGLProgram* GetGLProgram();

private:
	COGLProgram* m_pGLProgram;
};

#endif // _C_PROGRAM_H_