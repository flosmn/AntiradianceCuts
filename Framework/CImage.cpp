
#include "CImage.h"
#include "IL/il.h"
#include "IL/ilu.h"
#include "Defines.h"

#include <iostream>
#include <algorithm>

CImage::CImage(int width, int height)
	: m_Width(width), m_Height(height)
{
	ilInit();
	iluInit();
}

CImage::~CImage()
{
}

void CImage::LoadFromFile(const char* path, bool flipImage)
{
	ILuint handle;
	ilGenImages(1, &handle);
		
	ilBindImage(handle);
		
	ILboolean loaded = ilLoadImage(path);
	
	if(loaded == IL_FALSE)
	{		
		ilDeleteImages(1, &handle);

		return;
	}

	ILinfo ImageInfo;
	iluGetImageInfo(&ImageInfo);
	
	if(flipImage)
		iluFlipImage();

	//determine the file type
	ILenum fileType = ilGetInteger(IL_IMAGE_TYPE);
	ILenum fileFormat = ilGetInteger(IL_IMAGE_FORMAT);

	ILenum convertType = IL_FLOAT;
	ILenum convertFormat = IL_RGBA;

	// attempt to convert the image
	if (fileFormat != convertFormat || fileType != convertType)
	{
		loaded = ilConvertImage(convertFormat, convertType);
		if (loaded == false)
		{
			ilDeleteImages(1, &handle);

			std::cout << "Image not converted" << std::endl;
		
			return;
		}
	}
		
	iluGetImageInfo(&ImageInfo);
	ILuint w = ilGetInteger(IL_IMAGE_WIDTH);
	ILuint h = ilGetInteger(IL_IMAGE_HEIGHT);

	if (w != m_Width || h != m_Height)
	{
		iluImageParameter(ILU_FILTER, ILU_BILINEAR);
		ILboolean scaled = iluScale(m_Width, m_Height, 1);		
		if (scaled == false)
		{
			ilDeleteImages(1, &handle);

			std::cout << "Image not scaled" << std::endl;
		
			return;
		}
	}
		
	m_data.resize(m_Width * m_Height, glm::vec4(0.f));
	
	//copy data
	ilCopyPixels(0, 0, 0, m_Width, m_Height, 1, convertFormat, convertType, m_data.data());
	ILenum err = ilGetError();
	if (err != IL_NO_ERROR)
	{
		const char* err_str = iluErrorString(err);
		std::cout << "error copy pixels" << std::endl;
	}

	ilBindImage(0);
	ilDeleteImages(1, &handle);
}

void CImage::SetData(glm::vec4* pData)
{
	m_data.clear();
	m_data.resize(m_Width * m_Height);
	memcpy(m_data.data(), pData, m_Width * m_Height * sizeof(glm::vec4));
}

void CImage::GaussianBlur(int interations)
{
	ILuint handle;
	ilGenImages(1, &handle);
		
	ilBindImage(handle);

	ilTexImage(m_Width, m_Height, 1, 4,	IL_RGBA, IL_FLOAT, m_data.data());

	ILenum err = ilGetError();
	if (err != IL_NO_ERROR)
	{
		const char* err_str = iluErrorString(err);
		std::cout << "error tex image: " <<  err_str << std::endl;
	}

	iluBlurGaussian(interations);

	err = ilGetError();
	if (err != IL_NO_ERROR)
	{
		const char* err_str = iluErrorString(err);
		std::cout << "error gaussian blur: " <<  err_str << std::endl;
	}

	ilCopyPixels(0, 0, 0, m_Width, m_Height, 1, IL_RGBA, IL_FLOAT, m_data.data());
	err = ilGetError();
	if (err != IL_NO_ERROR)
	{
		const char* err_str = iluErrorString(err);
		std::cout << "error copy pixels" << std::endl;
	}
}

glm::vec4* CImage::GetData()
{
	return m_data.data();
}

void CImage::SaveAsPNG(const char* path, bool flipImage)
{
	ILuint handle;
	ilGenImages(1, &handle);
		
	ilBindImage(handle);

	unsigned char* pData = new unsigned char[3 * m_Width * m_Height];
	for(int i = 0; i < m_Width * m_Height; ++i)
	{
		pData[3 * i + 0] = (unsigned char)(std::min(std::max(255.f * m_data[i].x, 0.f), 255.f));
		pData[3 * i + 1] = (unsigned char)(std::min(std::max(255.f * m_data[i].y, 0.f), 255.f));
		pData[3 * i + 2] = (unsigned char)(std::min(std::max(255.f * m_data[i].z, 0.f), 255.f));
	}

	ilTexImage(m_Width, m_Height, 1, 3,	IL_RGB, IL_UNSIGNED_BYTE, pData);

	ILenum err = ilGetError();
	if (err != IL_NO_ERROR)
	{
		const char* err_str = iluErrorString(err);
		std::cout << "error tex image: " <<  err_str << std::endl;
	}

	ilEnable(IL_FILE_OVERWRITE);
	ILboolean saved = ilSave(IL_PNG, path);

	if(saved != IL_TRUE)
	{
		std::cout << "image not saved: " << std::endl;
	}

	err = ilGetError();
	if (err != IL_NO_ERROR)
	{
		const char* err_str = iluErrorString(err);
		std::cout << "error save image: " <<  err_str << std::endl;
	}

	delete [] pData;
}

void CImage::SaveAsHDR(const char* path, bool flipImage)
{
	ILuint handle;
	ilGenImages(1, &handle);
		
	ilBindImage(handle);

	ilTexImage(m_Width, m_Height, 1, 4,	IL_RGBA, IL_FLOAT, m_data.data());

	ILenum err = ilGetError();
	if (err != IL_NO_ERROR)
	{
		const char* err_str = iluErrorString(err);
		std::cout << "error tex image: " <<  err_str << std::endl;
	}

	ilEnable(IL_FILE_OVERWRITE);
	ILboolean saved = ilSave(IL_HDR, path);

	if(saved != IL_TRUE)
	{
		std::cout << "image not saved: " << std::endl;
	}

	err = ilGetError();
	if (err != IL_NO_ERROR)
	{
		const char* err_str = iluErrorString(err);
		std::cout << "error save image: " <<  err_str << std::endl;
	}
}