#ifndef _C_GL_RESOURCE_H_
#define _C_GL_RESOURCE_H_

#include <GL/glew.h>
#include <GL/gl.h>

#include <string>
#include <map>

#include "COGLBindLock.h"
#include "COGLBindSlot.h"

class COGLResource
{
	friend class COGLBindLock;

public:
	enum COGLResourceType { COGL_TEXTURE_2D, COGL_FRAMEBUFFER, COGL_RENDERBUFFER, COGL_VERTEXBUFFER,
		COGL_VERTEXARRAY, COGL_UNIFORMBUFFER, COGL_PROGRAM, COGL_SAMPLER, COGL_TEXTURE_BUFFER };

	COGLResource(COGLResourceType resourceType, std::string debugName);
	virtual ~COGLResource();

	virtual bool Init();
	virtual void Release();

	GLuint GetResourceIdentifier();
	std::string GetDebugName();

	bool CheckInitialized(std::string checker);
	bool CheckNotInitialized(std::string checker);
	bool CheckBound(std::string checker);
	bool CheckNotBound(std::string checker);
	bool CheckResourceNotNull(std::string checker);

protected:
	virtual void Bind(COGLBindSlot slot);
	virtual void Unbind();
	
	void PrintWarning(std::string warning);

	GLuint m_Resource;
	COGLBindSlot m_Slot;

	std::string m_DebugName;

private:
	std::map<COGLBindSlot, std::string>& GetMapForResourceType(COGLResourceType type);
	std::string GetStringForBindSlot(COGLBindSlot slot);

	COGLResourceType m_ResourceType;
	bool m_IsBound;
	bool m_IsInitialized;

	static std::map<COGLBindSlot, std::string> mapSlotsToCOGLTEXTURE2D;
	static std::map<COGLBindSlot, std::string> mapSlotsToCOGLTEXTUREBUFFER;
	static std::map<COGLBindSlot, std::string> mapSlotsToCOGLFRAMEBUFFER;
	static std::map<COGLBindSlot, std::string> mapSlotsToCOGLRENDERBUFFER;
	static std::map<COGLBindSlot, std::string> mapSlotsToCOGLVERTEXBUFFER;
	static std::map<COGLBindSlot, std::string> mapSlotsToCOGLVERTEXARRAY;
	static std::map<COGLBindSlot, std::string> mapSlotsToCOGLUNIFORMBUFFER;
	static std::map<COGLBindSlot, std::string> mapSlotsToCOGLPROGRAM;
};


#endif // _C_GL_RESOURCE_H_