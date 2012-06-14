#ifndef _C_OCL_RESOURCE_H_
#define _C_OCL_RESOURCE_H_

#include <string>

#include "..\Macros.h"

class COCLResource
{
public:
	COCLResource(std::string debugName);
	virtual ~COCLResource();

	virtual bool Init();
	virtual void Release();
		
	std::string GetDebugName();

	bool CheckInitialized(std::string checker);
	bool CheckNotInitialized(std::string checker);
	
protected:
	std::string m_DebugName;

private:
	bool m_IsInitialized;
};


#endif // _C_OCL_RESOURCE_H_