#ifndef _C_OGL_CONTEXT_H_
#define _C_OGL_CONTEXT_H_

#include <Windows.h>

class COGLContext
{
public:

	COGLContext(HGLRC context) : m_Context(context) { }
	~COGLContext() { }

	HGLRC GetGLContext() { return m_Context; }

private:
	HGLRC m_Context;
};

#endif // _C_OGL_CONTEXT_H_