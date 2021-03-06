#include "CExport.h"

typedef unsigned int uint;

#include "..\OGLResources\COGLTexture2D.h"

#include "..\CImage.h"

#include <iostream>
#include <fstream>
#include <assert.h>
#include <sstream>

CExport::CExport()
{
}

CExport::~CExport()
{
}

void CExport::ExportPFM(COGLTexture2D* pTexture, std::string strFileName)
{
	uint width = pTexture->GetWidth();
	uint height = pTexture->GetHeight();

	// get the pixel data of the texture. assertion is that the format of the texture is 32F, RGBA format
	assert(pTexture->GetInternalFormat() == GL_RGBA32F);

	float* pTextureData = new float[4 * width * height];
	pTexture->GetPixelData(pTextureData);

	float* pFloatMap = new float[3 * width * height];
	memset(pFloatMap, 0, 3 * width * height * sizeof(float));

	for(uint i = 0; i < width * height; ++i)
	{
		pFloatMap[3 * i + 0] = pTextureData[4 * i + 0];
		pFloatMap[3 * i + 1] = pTextureData[4 * i + 1];
		pFloatMap[3 * i + 2] = pTextureData[4 * i + 2];
	}
	
	char* pPixelData = new char[4 * 3 * width * height];
	memcpy(pPixelData, pFloatMap, 4 * 3 * width * height);
	
	std::string file("Export\\");
	file = file.append(strFileName);

	std::ofstream os;
	os.open(file, std::ios::out | std::ios::binary);
	
	// pfm header
	os << "PF\n";
	os << width << " " << height <<"\n";
	os << "-1.0" << "\n";

	// pfm data
	os.write(pPixelData, 4 * 3 * width * height);

	os.close();

	delete [] pTextureData;
	delete [] pFloatMap;
	delete [] pPixelData;
}

void CExport::ExportPNG(COGLTexture2D* pTexture, std::string strFileName)
{
	uint width = pTexture->GetWidth();
	uint height = pTexture->GetHeight();

	// get the pixel data of the texture. assertion is that the format of the texture is 32F, RGBA format
	assert(pTexture->GetInternalFormat() == GL_RGBA32F);

	glm::vec4* pTextureData = new glm::vec4[width * height];
	pTexture->GetPixelData(pTextureData);

	CImage image(width, height);
	image.SetData(pTextureData);
	
	std::stringstream ss;
	ss << "Export/" << strFileName;
	image.SaveAsPNG(ss.str().c_str(), false);

	delete [] pTextureData;
}

void CExport::ExportHDR(COGLTexture2D* pTexture, std::string strFileName)
{
	uint width = pTexture->GetWidth();
	uint height = pTexture->GetHeight();

	// get the pixel data of the texture. assertion is that the format of the texture is 32F, RGBA format
	assert(pTexture->GetInternalFormat() == GL_RGBA32F);

	glm::vec4* pTextureData = new glm::vec4[width * height];
	pTexture->GetPixelData(pTextureData);

	CImage image(width, height);
	image.SetData(pTextureData);
	
	std::stringstream ss;
	ss << "Export/" << strFileName;
	image.SaveAsHDR(ss.str().c_str(), false);

	delete [] pTextureData;
}