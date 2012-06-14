#ifndef _C_SHADOW_MAP_H_
#define _C_SHADOW_MAP_H_

typedef unsigned int uint; 

#include "GL/glew.h"

class COGLTexture2D;
class COGLFrameBuffer;

class CShadowMap 
{
public:
	CShadowMap();
	~CShadowMap();

	bool Init(uint size);
	void Release();

	COGLTexture2D* GetShadowMapTexture() { return m_pShadowMapTexture; }
	COGLFrameBuffer* GetRenderTarget() { return m_pRenderTarget; }

	uint GetShadowMapSize() { return m_ShadowMapSize; }

private:
	COGLFrameBuffer* m_pRenderTarget;
	COGLTexture2D* m_pShadowMapTexture;

	uint m_ShadowMapSize;
};

#endif // _C_SHADOW_MAP_H_