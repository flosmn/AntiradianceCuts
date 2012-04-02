#ifndef _C_PROGRAM_H_
#define _C_PROGRAM_H_

typedef unsigned int uint;

#include <string>

class CGLProgram;
class CGLUniformBuffer;
class CGLSampler;

class CProgram
{
public:
	CProgram(const std::string debugName, const std::string VS, const std::string FS);
	~CProgram();

	bool Init();
	void Release();

	void BindUniformBuffer(CGLUniformBuffer* pGLUniformBuffer, std::string strUniformBlockNam);
	void BindSampler(uint samplerSlot, CGLSampler* pGLSampler);

	CGLProgram* GetGLProgram();

private:
	CGLProgram* m_pGLProgram;
};

#endif // _C_PROGRAM_H_