#ifndef _C_SHADOW_MAP_H_
#define _C_SHADOW_MAP_H_

typedef unsigned int uint; 

#include "GL/glew.h"

class CGLTexture2D;
class CGLFrameBuffer;

class CShadowMap 
{
public:
	CShadowMap();
	~CShadowMap();

	bool Init(uint size);
	void Release();

	CGLTexture2D* GetShadowMapTexture() { return m_pShadowMapTexture; }
	CGLFrameBuffer* GetRenderTarget() { return m_pRenderTarget; }

	uint GetShadowMapSize() { return m_ShadowMapSize; }

private:
	CGLFrameBuffer* m_pRenderTarget;
	CGLTexture2D* m_pShadowMapTexture;

	uint m_ShadowMapSize;
};

#endif // _C_SHADOW_MAP_H_