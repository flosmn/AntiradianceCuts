#ifndef _C_OCL_PROGRAM_H_
#define _C_OCL_PROGRAM_H_

#include "CL/cl.h"

#include "COCLResource.h"

#include <string>

class COCLContext;

class COCLProgram : public COCLResource
{
public:
	COCLProgram(COCLContext* pContext, const std::string& debugName);
	~COCLProgram();

	virtual bool Init(const std::string& source);
	virtual void Release();

	cl_program* GetCLProgram() { CheckInitialized("COCLProgram.GetCLProgram()"); return &m_Program; }

private:
	cl_program m_Program;

	COCLContext* m_pContext;	
};

#endif // _C_OCL_PROGRAM_H_