#include "..\CImage.h"
#include "COGLCubeMap.h"

#include "..\Macros.h"
#include "..\Utils\GLErrorUtil.h"

GLenum cube[6] = { 
	GL_TEXTURE_CUBE_MAP_POSITIVE_X, 
	GL_TEXTURE_CUBE_MAP_NEGATIVE_X,
	GL_TEXTURE_CUBE_MAP_POSITIVE_Y, 
	GL_TEXTURE_CUBE_MAP_NEGATIVE_Y,
	GL_TEXTURE_CUBE_MAP_POSITIVE_Z, 
	GL_TEXTURE_CUBE_MAP_NEGATIVE_Z,
};

COGLCubeMap::COGLCubeMap(const char* debugName)
	: COGLResource(COGL_CUBE_MAP, debugName)
{
}

COGLCubeMap::~COGLCubeMap()
{
	COGLResource::~COGLResource();
}

bool COGLCubeMap::Init(unsigned int width, unsigned int height, GLenum internalFormat, unsigned int nMipMapLevels)
{
	V_RET_FOF(COGLResource::Init());

	glGenTextures(1, &m_Resource);

	m_Width = width;
	m_Height = height;
	m_InternalFormat = internalFormat;
	
	COGLBindLock lock(this, COGL_TEXTURE0_SLOT);

	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAX_LEVEL, nMipMapLevels);
	glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_GENERATE_MIPMAP, true);
	
	return true;
}

void COGLCubeMap::Release()
{
	COGLResource::Release();
	
	glDeleteTextures(1, &m_Resource);
}

void COGLCubeMap::LoadCubeMapFromFiles(
	const char* pos_x, const char* neg_x, 
	const char* pos_y, const char* neg_y,
	const char* pos_z, const char* neg_z)
{
	CheckInitialized("COGLCubeMap::LoadCubeMapFromFiles()");
	CheckResourceNotNull("COGLCubeMap::LoadCubeMapFromFiles()");
		
	const char* paths[6] = { pos_x, neg_x, pos_y, neg_y, pos_z, neg_z };
	
	COGLBindLock lock(this, COGL_TEXTURE0_SLOT);
	
	for (int i = 0; i < 6; ++i)
	{
		CImage* image = new CImage(m_Width, m_Height);
		image->LoadFromFile(paths[i], false);

		glm::vec4* pData = image->GetData();
		
		glTexImage2D(cube[i], 0, m_InternalFormat, m_Width, m_Height, 0, GL_RGBA, GL_FLOAT, pData);	

		delete image;
	}
	glGenerateMipmap(GL_TEXTURE_CUBE_MAP);	
}

void COGLCubeMap::LoadCubeMapFromPath(const std::string& path)
{
	std::string p1(path);
	std::string p2(path);
	std::string p3(path);
	std::string p4(path);
	std::string p5(path);
	std::string p6(path);

	LoadCubeMapFromFiles(
		p1.append("pos_x.png").c_str(), 
		p2.append("neg_x.png").c_str(),
		p3.append("pos_y.png").c_str(),
		p4.append("neg_y.png").c_str(),
		p5.append("pos_z.png").c_str(),
		p6.append("neg_z.png").c_str());
}

void COGLCubeMap::Bind(COGLBindSlot slot)
{
	COGLResource::Bind(slot);

	glActiveTexture(GetGLSlot(slot));

	glBindTexture(GL_TEXTURE_CUBE_MAP, m_Resource);
}

void COGLCubeMap::Unbind()
{
	COGLResource::Unbind();

	glBindTexture(GL_TEXTURE_CUBE_MAP, 0);
}

GLuint COGLCubeMap::GetResourceIdentifier()
{
	CheckInitialized("COGLCubeMap::GetResourceIdentifier");
	CheckResourceNotNull("COGLCubeMap::GetResourceIdentifier");

	return m_Resource;
}