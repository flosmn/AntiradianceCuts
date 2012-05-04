#ifndef _C_GL_RESOURCE_H_
#define _C_GL_RESOURCE_H_

#include <GL/glew.h>
#include <GL/gl.h>

#include <string>
#include <map>

#include "CGLBindLock.h"
#include "CGLBindSlot.h"

class CGLResource
{
	friend class CGLBindLock;

public:
	enum CGLResourceType { CGL_TEXTURE_2D, CGL_FRAMEBUFFER, CGL_RENDERBUFFER, CGL_VERTEXBUFFER,
		CGL_VERTEXARRAY, CGL_UNIFORMBUFFER, CGL_PROGRAM, CGL_SAMPLER, CGL_TEXTURE_BUFFER };

	CGLResource(CGLResourceType resourceType, std::string debugName);
	virtual ~CGLResource();

	virtual bool Init();
	virtual void Release();

	GLuint GetResourceIdentifier();
	std::string GetDebugName();

protected:
	virtual void Bind(CGLBindSlot slot);
	virtual void Unbind();

	bool CheckInitialized(std::string checker);
	bool CheckNotInitialized(std::string checker);
	bool CheckBound(std::string checker);
	bool CheckNotBound(std::string checker);
	bool CheckResourceNotNull(std::string checker);

	void PrintWarning(std::string warning);

	GLuint m_Resource;
	CGLBindSlot m_Slot;

	std::string m_DebugName;

private:
	std::map<CGLBindSlot, std::string>& GetMapForResourceType(CGLResourceType type);
	std::string GetStringForBindSlot(CGLBindSlot slot);

	CGLResourceType m_ResourceType;
	bool m_IsBound;
	bool m_IsInitialized;

	static std::map<CGLBindSlot, std::string> mapSlotsToCGLTEXTURE2D;
	static std::map<CGLBindSlot, std::string> mapSlotsToCGLTEXTUREBUFFER;
	static std::map<CGLBindSlot, std::string> mapSlotsToCGLFRAMEBUFFER;
	static std::map<CGLBindSlot, std::string> mapSlotsToCGLRENDERBUFFER;
	static std::map<CGLBindSlot, std::string> mapSlotsToCGLVERTEXBUFFER;
	static std::map<CGLBindSlot, std::string> mapSlotsToCGLVERTEXARRAY;
	static std::map<CGLBindSlot, std::string> mapSlotsToCGLUNIFORMBUFFER;
	static std::map<CGLBindSlot, std::string> mapSlotsToCGLPROGRAM;
};


#endif // _C_GL_RESOURCE_H_