#ifndef _C_OGL_CUBE_MAP_H_
#define _C_OGL_CUBE_MAP_H_

#include "GL/glew.h"
#include "GL/gl.h"

#include "COGLResource.h"
#include "COGLBindSlot.h"

#include <string>

class COGLCubeMap : public COGLResource
{
public:
	COGLCubeMap(unsigned int width, unsigned int height, GLenum internalFormat, unsigned int nMipMapLevels, std::string const& debugName = "");
	~COGLCubeMap();

	void LoadCubeMapFromFiles(
		const char* pos_x, const char* neg_x, 
		const char* pos_y, const char* neg_y,
		const char* pos_z, const char* neg_z);
	
	void LoadCubeMapFromPath(const std::string& path);
	
	virtual GLuint GetResourceIdentifier();

private:
	virtual void Bind(COGLBindSlot slot);
	virtual void Unbind();
	
	int m_Width;
	int m_Height;
	GLenum m_InternalFormat;
};

#endif // _C_OGL_CUBE_MAP_H_