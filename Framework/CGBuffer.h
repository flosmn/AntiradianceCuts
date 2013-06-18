#ifndef GBUFFER_H
#define GBUFFER_H

#include <memory>

typedef unsigned int uint;

class Scene;
class ShadowMap;
class Light;

class COGLTexture2D;
class COGLFrameBuffer;
class COGLSampler;
class CFullScreenQuad;
class CProgram;
class CShadowMap;

class CGBuffer
{
public:
	CGBuffer(uint width, uint height, COGLTexture2D* pDepthBuffer);
	~CGBuffer();

	COGLTexture2D* GetPositionTextureWS() { return m_positionWS.get(); }
	COGLTexture2D* GetNormalTexture() { return m_normalWS.get(); }
	COGLTexture2D* GetMaterialTexture() { return m_materials.get(); }
	COGLFrameBuffer* GetRenderTarget() { return m_renderTarget.get(); }

private:
	uint m_Width;
	uint m_Height;

	std::unique_ptr<COGLFrameBuffer> m_renderTarget;
	std::unique_ptr<COGLTexture2D> m_positionWS;
	std::unique_ptr<COGLTexture2D> m_normalWS;
	std::unique_ptr<COGLTexture2D> m_materials;
	std::unique_ptr<COGLSampler> m_pointSampler;

	std::unique_ptr<CProgram> m_program;
};

#endif