
#include "CImage.h"
#include "IL/il.h"
#include "IL/ilu.h"
#include "Defines.h"

#include <iostream>

CImage::CImage(int width, int height)
	: m_Width(width), m_Height(height), m_pData(nullptr)
{
	ilInit();
	iluInit();
}

CImage::~CImage()
{
	SAFE_DELETE_ARRAY(m_pData);
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
		
	m_pData = new glm::vec4[m_Width * m_Height];

	//copy data
	ilCopyPixels(0, 0, 0, m_Width, m_Height, 1, convertFormat, convertType, m_pData);
	ILenum err = ilGetError();
	if (err != IL_NO_ERROR)
	{
		const char* err_str = iluErrorString(err);
		std::cout << "error copy pixels" << std::endl;
	}

	ilBindImage(0);
	ilDeleteImages(1, &handle);
}

glm::vec4* CImage::GetData()
{
	return m_pData;
}