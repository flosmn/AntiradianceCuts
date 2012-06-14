#ifndef GBUFFER_H
#define GBUFFER_H

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
	CGBuffer();
	~CGBuffer();

	bool Init(uint width, uint height, COGLTexture2D* pDepthBuffer);
	void Release();

	COGLTexture2D* GetPositionTextureWS() { return m_pGLTTexturePositionWS; }
	COGLTexture2D* GetNormalTexture() { return m_pGLTTextureNormalWS; }
	COGLTexture2D* GetMaterialTexture() { return m_pGLTTextureMaterials; }
	COGLFrameBuffer* GetRenderTarget() { return m_pGLFBRenderTarget; }

private:
	uint m_Width;
	uint m_Height;

	COGLFrameBuffer* m_pGLFBRenderTarget;
	COGLTexture2D* m_pGLTTexturePositionWS;
	COGLTexture2D* m_pGLTTextureNormalWS;
	COGLTexture2D* m_pGLTTextureMaterials;
	COGLSampler* m_pGLPointSampler;

	CProgram* m_pCreateGBufferProgram;
};

#endif