#ifndef _OCL_UTIL_H_
#define _OCL_UTIL_H_

#include "CL/cl.h"

#include <string>

bool CHECK_CL_SUCCESS(cl_uint err, std::string message);

std::string ReadKernelFile(const std::string& strKernelFilename);
std::string FindFileOrThrow(const std::string &strBasename);
std::string GetErrorString(cl_uint err);

#endif _OCL_UTIL_H_