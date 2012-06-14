#ifndef _C_ACCUMULATION_BUFFER_H
#define _C_ACCUMULATION_BUFFER_H

typedef unsigned int uint;

class COGLTexture2D;
class COGLFrameBuffer;

class CAccumulationBuffer
{
public:
	CAccumulationBuffer();
	~CAccumulationBuffer();
	
	bool Init(uint width, uint height, COGLTexture2D* pDepthBuffer);
	void Release();

	COGLTexture2D* GetTexture() { return m_pGLTAccumTexture; }
	COGLFrameBuffer* GetRenderTarget() { return m_pGLFBRenderTarget; }
	
private:
	COGLFrameBuffer* m_pGLFBRenderTarget;
	COGLTexture2D* m_pGLTAccumTexture;
	COGLTexture2D* m_pDepthBuffer;

	bool m_ExternalDepthBuffer;

	uint m_Width, m_Height;
};

#endif // _C_ACCUMULATION_BUFFER_H