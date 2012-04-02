#ifndef _C_GL_BIND_LOCK_H_
#define _C_GL_BIND_LOCK_H_

#include <GL/glew.h>
#include <GL/GL.h>

#include "CGLBindSlot.h"

class CGLResource;

class CGLBindLock
{
public:
	CGLBindLock(CGLResource* resource, CGLBindSlot slot);
	~CGLBindLock();

private:
	CGLResource* m_pResource;

	// noncopyable
    CGLBindLock(const CGLBindLock&);
    CGLBindLock& operator=(const CGLBindLock&);
};

#endif _C_GL_BIND_LOCK_H_