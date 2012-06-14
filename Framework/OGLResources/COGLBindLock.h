#ifndef _C_GL_BIND_LOCK_H_
#define _C_GL_BIND_LOCK_H_

#include <GL/glew.h>
#include <GL/GL.h>

#include "COGLBindSlot.h"

class COGLResource;

class COGLBindLock
{
public:
	COGLBindLock(COGLResource* resource, COGLBindSlot slot);
	~COGLBindLock();

private:
	COGLResource* m_pResource;

	// noncopyable
    COGLBindLock(const COGLBindLock&);
    COGLBindLock& operator=(const COGLBindLock&);
};

#endif _C_GL_BIND_LOCK_H_