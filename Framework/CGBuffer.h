#ifndef GBUFFER_H
#define GBUFFER_H

typedef unsigned int uint;

class Scene;
class ShadowMap;
class Light;

class CGLTexture2D;
class CGLFrameBuffer;
class CGLSampler;
class CFullScreenQuad;
class CProgram;
class CShadowMap;

class CGBuffer
{
public:
	CGBuffer();
	~CGBuffer();

	bool Init(uint width, uint height, CGLTexture2D* pDepthBuffer);
	void Release();

	CGLTexture2D* GetPositionTextureWS() { return m_pGLTTexturePositionWS; }
	CGLTexture2D* GetNormalTexture() { return m_pGLTTextureNormalWS; }
	CGLTexture2D* GetMaterialTexture() { return m_pGLTTextureMaterials; }
	CGLFrameBuffer* GetRenderTarget() { return m_pGLFBRenderTarget; }

private:
	uint m_Width;
	uint m_Height;

	CGLFrameBuffer* m_pGLFBRenderTarget;
	CGLTexture2D* m_pGLTTexturePositionWS;
	CGLTexture2D* m_pGLTTextureNormalWS;
	CGLTexture2D* m_pGLTTextureMaterials;
	CGLSampler* m_pGLPointSampler;

	CProgram* m_pCreateGBufferProgram;
};

#endif