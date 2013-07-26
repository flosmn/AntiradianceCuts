#ifndef GL_ERRORUTIL_H_
#define GL_ERRORUTIL_H_

#include <GL/glew.h>
#include <GL/gl.h>

#include <string>
#include <iostream>

static bool CheckGLError(std::string checker, std::string location) {
	GLenum err = glGetError();
	if (err != GL_NO_ERROR)
	{
		std::cout << checker << " has found an error in " << location 
			<< ". GLError: " << gluErrorString(err) << std::endl;
		return true;
	}

	return false;
}

#endif // GL_ERRORUTIL_H_