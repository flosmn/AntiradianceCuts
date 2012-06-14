#include "COCLContext.h"

#include "CL/cl_gl.h"

#include "OCLUtil.h"

#include "..\OGLResources\COGLContext.h"

#include <string>
#include <iostream>

typedef unsigned int uint;

COCLContext::COCLContext()
	: COCLResource("COCLContext")
{
}

COCLContext::~COCLContext()
{
	CheckNotInitialized("COCLContext.~COCLContext()");
}

bool COCLContext::Init()
{
	cl_int err;

	V_RET_FOF(GetPlatform());
	V_RET_FOF(GetDevice());

	m_Context = clCreateContext(NULL, 1, &m_DeviceId, NULL, NULL, &err);

	V_RET_FOF(CHECK_CL_SUCCESS(err, "clCreateContext"));

	m_CommandQueue = clCreateCommandQueue(m_Context, m_DeviceId, NULL, &err);

	V_RET_FOF(CHECK_CL_SUCCESS(err, "clCreateCommandQueue"));

	V_RET_FOF(COCLResource::Init());

	return true;
}

bool COCLContext::Init(COGLContext* pGLContext)
{
	cl_int err;

	V_RET_FOF(GetPlatform());
	V_RET_FOF(GetDevice());
	
	const char* CL_GL_SHARING_EXT = "cl_khr_gl_sharing";
			
	// Get string containing supported device extensions
	size_t ext_size = 1024;
	char* ext_string = (char*)malloc(ext_size);
	err = clGetDeviceInfo(m_DeviceId, CL_DEVICE_EXTENSIONS, ext_size, ext_string, &ext_size);

	// Search for GL support in extension string (space delimited)
	int supported = IsExtensionSupported(CL_GL_SHARING_EXT, ext_string, ext_size);
	if( ! supported )
	{
		return false;
	}

	// Create CL context properties, add WGL context & handle to DC
	cl_context_properties properties[] = {
		CL_GL_CONTEXT_KHR, (cl_context_properties)wglGetCurrentContext(), // WGL Context
		CL_WGL_HDC_KHR, (cl_context_properties)wglGetCurrentDC(),		  // WGL HDC
		CL_CONTEXT_PLATFORM, (cl_context_properties)m_PlatformId,		  // OpenCL platform
		0
	};

	typedef CL_API_ENTRY cl_int (CL_API_CALL *P1)(
		const cl_context_properties *properties,
        cl_gl_context_info param_name,
        size_t param_value_size,
        void *param_value,
        size_t *param_value_size_ret);

	CL_API_ENTRY cl_int (CL_API_CALL *myclGetGLContextInfoKHR)(
		const cl_context_properties *properties,
        cl_gl_context_info param_name,
        size_t param_value_size,
        void *param_value,
        size_t *param_value_size_ret)=NULL;

    myclGetGLContextInfoKHR=(P1)clGetExtensionFunctionAddress("clGetGLContextInfoKHR");
    
	// Find CL capable devices in the current GL context
	cl_device_id devices[1]; size_t size;
		
	myclGetGLContextInfoKHR(properties, CL_DEVICES_FOR_GL_CONTEXT_KHR, 1 * sizeof(cl_device_id), devices, &size);
	// Create a context using the supported devices
	int count = size / sizeof(cl_device_id);
	m_Context = clCreateContext(properties, count, devices, NULL, 0, &err);

	V_RET_FOF(CHECK_CL_SUCCESS(err, "clCreateContext()"));

	m_CommandQueue = clCreateCommandQueue(m_Context, m_DeviceId, NULL, &err);

	V_RET_FOF(CHECK_CL_SUCCESS(err, "clCreateCommandQueue"));

	V_RET_FOF(COCLResource::Init());

	return true;
}

void COCLContext::Release()
{
	COCLResource::Release();

	CHECK_CL_SUCCESS(clReleaseCommandQueue(m_CommandQueue), "clReleaseCommandQueue");

	CHECK_CL_SUCCESS(clReleaseContext(m_Context), "clReleaseContext");
}

bool COCLContext::GetPlatform()
{
	cl_uint nPlatforms = 0;
	cl_platform_id* platforms = new cl_platform_id[2];

	V_RET_FOF(CHECK_CL_SUCCESS(clGetPlatformIDs(2, platforms, &nPlatforms), "clGetPlatformIDs"));
	
	if(nPlatforms == 0) return false;

	char buffer[4096];
	V_RET_FOF(CHECK_CL_SUCCESS(clGetPlatformInfo(platforms[0], CL_PLATFORM_VERSION, 4096, buffer, NULL), "clGetPlatformInfo"));
	std::cout << "CL_PLATFORM_VERSION: " << buffer << std::endl;
	V_RET_FOF(CHECK_CL_SUCCESS(clGetPlatformInfo(platforms[0], CL_PLATFORM_NAME, 4096, buffer, NULL), "clGetPlatformInfo"));
	std::cout << "CL_PLATFORM_NAME: " << buffer << std::endl;
	V_RET_FOF(CHECK_CL_SUCCESS(clGetPlatformInfo(platforms[0], CL_PLATFORM_VENDOR, 4096, buffer, NULL), "clGetPlatformInfo"));
	std::cout << "CL_PLATFORM_VENDOR: " << buffer << std::endl;

	m_PlatformId = platforms[0];

	return true;
}

bool COCLContext::GetDevice()
{
	cl_uint nDevices = 0;
	
	cl_device_id* devices = new cl_device_id[2];
	
	V_RET_FOF(CHECK_CL_SUCCESS(clGetDeviceIDs(m_PlatformId, CL_DEVICE_TYPE_GPU, 2, devices, &nDevices), "clGetDeviceIDs"));

	if(nDevices == 0) return false;
	
	m_DeviceId = devices[0];

	char buffer[4096];
	V_RET_FOF(CHECK_CL_SUCCESS(clGetDeviceInfo(m_DeviceId, CL_DEVICE_NAME, 4096, buffer, NULL), "clGetDeviceInfo"));
	std::cout << "CL_DEVICE_NAME: " << buffer << std::endl;
	V_RET_FOF(CHECK_CL_SUCCESS(clGetDeviceInfo(m_DeviceId, CL_DEVICE_VERSION, 4096, buffer, NULL), "clGetDeviceInfo"));
	std::cout << "CL_DEVICE_VERSION: " << buffer << std::endl;
	V_RET_FOF(CHECK_CL_SUCCESS(clGetDeviceInfo(m_DeviceId, CL_DRIVER_VERSION, 4096, buffer, NULL), "clGetDeviceInfo"));
	std::cout << "CL_DRIVER_VERSION: " << buffer << std::endl;

	return true;
}

bool COCLContext::IsExtensionSupported(const char* support_str, const char* ext_string, size_t ext_buffer_size)
{
	std::string support(support_str);
	std::string ext(ext_string);

	if(ext.find(support) == std::string::npos)
	{
		std::cout << "Extension not supported: " << support << std::endl;
		return false;
	}
	else {
		std::cout << "Info: Found extension support " << support << std::endl;
		return true;
	}
}