#ifndef _C_ACCUMULATION_BUFFER_H
#define _C_ACCUMULATION_BUFFER_H

#include <memory>

typedef unsigned int uint;

class COGLTexture2D;
class COGLFrameBuffer;

class CAccumulationBuffer
{
public:
	CAccumulationBuffer(uint width, uint height, COGLTexture2D* pDepthBuffer = 0);
	~CAccumulationBuffer();
	
	COGLTexture2D* GetTexture() { return m_accumTexture.get(); }
	COGLFrameBuffer* GetRenderTarget() { return m_renderTarget.get(); }
	
private:
	std::unique_ptr<COGLFrameBuffer> m_renderTarget;
	std::unique_ptr<COGLTexture2D> m_accumTexture;
	std::unique_ptr<COGLTexture2D> m_depthBuffer;

	bool m_ExternalDepthBuffer;

	uint m_Width, m_Height;
};

#endif // _C_ACCUMULATION_BUFFER_H