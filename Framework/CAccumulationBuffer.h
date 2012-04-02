#ifndef _C_ACCUMULATION_BUFFER_H
#define _C_ACCUMULATION_BUFFER_H

typedef unsigned int uint;

class CGLTexture2D;
class CGLRenderBuffer;
class CGLFrameBuffer;

class CAccumulationBuffer
{
public:
	CAccumulationBuffer();
	~CAccumulationBuffer();
	
	bool Init(uint width, uint height);
	void Release();

	CGLTexture2D* GetTexture() { return m_pGLTAccumTexture; }
	CGLFrameBuffer* GetRenderTarget() { return m_pGLFBRenderTarget; }
	
private:
	CGLFrameBuffer* m_pGLFBRenderTarget;
	CGLTexture2D* m_pGLTAccumTexture;
	CGLRenderBuffer* m_pGLRBDepthRenderBuffer;

	uint m_Width, m_Height;
};

#endif // _C_ACCUMULATION_BUFFER_H