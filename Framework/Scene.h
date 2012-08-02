#ifndef SCENE_H
#define SCENE_H

typedef unsigned int uint;

#include "GL/glew.h"

#include "Ray.h"
#include "Intersection.h"

#include "Utils\mtrand.h"

#include <vector>

class CCamera;
class AVPL;
class AreaLight;
class CKdTreeAccelerator;
class CConfigManager;
class CAVPLImportanceSampling;

class COGLUniformBuffer;

class CModel;

class Scene
{
public:
	Scene(CCamera* pCamera, CConfigManager* pConfManager);
	~Scene();
	
	bool Init();
	void Release();
	
	void ClearScene();
	
	void DrawScene(COGLUniformBuffer* pUBTransform, COGLUniformBuffer* pUBMaterial);
	void DrawScene(COGLUniformBuffer* pUBTransform);

	void DrawScene(const glm::mat4& mView, const glm::mat4& mProj, COGLUniformBuffer* pUBTransform);

	void DrawAreaLight(COGLUniformBuffer* pUBTransform, COGLUniformBuffer* pUBAreaLight);

	void UpdateAreaLights();

	void LoadCornellBox();
	void LoadCornellBoxSmall();
	void LoadSibernik();
	void LoadSimpleScene();
	void LoadCornellBoxDragon();
		
	bool IntersectRayScene(const Ray& ray, float* t, Intersection *pIntersection);
	bool IntersectRaySceneSimple(const Ray& ray, float* t, Intersection *pIntersection);

	CCamera* GetCamera() { return m_Camera; }

	void ClearLighting();
	void SetAVPLImportanceSampling(CAVPLImportanceSampling *pAVPLIS) { m_pAVPLImportanceSampling = pAVPLIS; }

	int GetNumberOfLightPaths() { return m_NumLightPaths; }
				
	void CreatePaths(std::vector<AVPL*>& avpls, uint numPaths, int N, int nAdditionalAVPLs);
	void CreatePath(std::vector<AVPL*>& avpls, int N, int nAdditionalAVPLs);
	void CreatePrimaryVpls(std::vector<AVPL*>& avpls, int numVpls);

	uint GetNumCreatedAVPLs() { return m_NumCreatedAVPLs; }
	uint GetNumAVPLsAfterIS() { return m_NumAVPLsAfterIS; }

private:
	void ClearPath();
		
	void InitKdTree();
	void ReleaseKdTree();

	AVPL* CreateAVPL(AVPL* pred, int N, int nAdditionalAVPLs);
	AVPL* ContinueAVPLPath(AVPL* pred, glm::vec3 direction, float pdf, int N, int nAdditionalAVPLs);
	void CreateAVPLs(AVPL* pred, std::vector<AVPL*>& path, int N, int nAVPLs);
	
	std::vector<CModel*> m_Models;
		
	int m_CurrentBounce;
	int m_NumLightPaths;

	CCamera* m_Camera;		
	AreaLight* m_AreaLight;

	CKdTreeAccelerator* m_pKdTreeAccelerator;
	std::vector<CPrimitive*> m_Primitives;

	uint m_NumCreatedAVPLs;
	uint m_NumAVPLsAfterIS;

	CConfigManager* m_pConfManager;
	CAVPLImportanceSampling* m_pAVPLImportanceSampling;
};

#endif SCENE_H