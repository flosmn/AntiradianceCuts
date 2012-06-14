#include "OCLUtil.h"

#include <iostream>
#include <istream>
#include <fstream>
#include <sstream>

bool CHECK_CL_SUCCESS(cl_uint err, std::string message)
{
	if(err == CL_SUCCESS)
	{
		return true;
	}
	else
	{
		std::cout << "CL error detected: " << err << ". message: " << message << std::endl;
		return false;
	}
}

std::string ReadKernelFile(const std::string& strKernelFilename)
{
	std::string strFilename = FindFileOrThrow(strKernelFilename);
	std::ifstream kernelFile(strFilename.c_str());
	std::stringstream kernelData;
	kernelData << kernelFile.rdbuf();
	kernelFile.close();

	return kernelData.str();
}

std::string FindFileOrThrow(const std::string &strBasename)
{
	std::string strFilename = strBasename;
	std::ifstream testFile(strFilename.c_str());
	if(testFile.is_open())
		return strFilename;
	
	throw std::runtime_error("Could not find the file " + strBasename);
}

