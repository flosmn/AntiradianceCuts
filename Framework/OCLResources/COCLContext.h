#ifndef _C_OCL_CONTEXT_H_
#define _C_OCL_CONTEXT_H_

#include "CL/cl.h"

#include "COCLResource.h"

class COGLContext;

class COCLContext : public COCLResource
{
public:
	COCLContext(COGLContext* pOGLContext);
	~COCLContext();
	
	const cl_context* GetCLContext() { CheckInitialized("COCLContext.GetCLContext()"); return &m_Context; }
	const cl_device_id* GetCLDeviceId() { CheckInitialized("COCLContext.GetCLDeviceId()"); return &m_DeviceId; }
	const cl_command_queue* GetCLCommandQueue() { CheckInitialized("COCLContext.GetCLCommandQueue()"); return &m_CommandQueue; } 

	size_t GetMaxWorkGroupSize() { CheckInitialized("COCLContext.GetMaxWorkGroupSize()"); return m_MaxWorkGroupSize; }
	size_t* GetMaxWorkGroupDimensions2DSquare() { CheckInitialized("COCLContext.GetMaxWorkGroupDimensions2DSquare()"); return m_MaxWorkGroupDimensions2DSquare; }

private:
	bool GetPlatform();
	bool GetDevice();
	bool CreateCommandQueue();

	bool IsExtensionSupported(const char* support_str, const char* ext_string, size_t ext_buffer_size);

	cl_platform_id m_PlatformId;
	cl_device_id m_DeviceId;
	cl_context m_Context;
	cl_command_queue m_CommandQueue;

	COGLContext* m_pOGLContext;

	size_t m_MaxWorkGroupSize;
	size_t m_MaxWorkItemSizes[3];
	size_t m_MaxWorkGroupDimensions2DSquare[2];
};

#endif // _C_OCL_CONTEXT_H_