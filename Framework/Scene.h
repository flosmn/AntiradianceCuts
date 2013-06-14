#ifndef SCENE_H
#define SCENE_H

typedef unsigned int uint;

#include "GL/glew.h"

#include "Ray.h"
#include "Intersection.h"
#include "CPrimitive.h"
#include "Material.h"

#include "Utils\mtrand.h"

#include <vector>

class CCamera;
class AVPL;
class AreaLight;
class CKdTreeAccelerator;
class CConfigManager;
class CAVPLImportanceSampling;
class CMaterialBuffer;
class CReferenceImage;

class COGLUniformBuffer;
class COCLContext;

class CModel;

class Scene
{
public:
	Scene(CCamera* pCamera, CConfigManager* pConfManager, COCLContext* pOCLContext);
	~Scene();
	
	bool Init();
	void Release();
	
	void ClearScene();
	
	void DrawScene(COGLUniformBuffer* pUBTransform, COGLUniformBuffer* pUBMaterial);
	void DrawScene(COGLUniformBuffer* pUBTransform);

	void DrawScene(const glm::mat4& mView, const glm::mat4& mProj, COGLUniformBuffer* pUBTransform);

	void DrawAreaLight(COGLUniformBuffer* pUBTransform, COGLUniformBuffer* pUBAreaLight);
	void DrawAreaLight(COGLUniformBuffer* pUBTransform, COGLUniformBuffer* pUBAreaLight, glm::vec3 color);

	void UpdateAreaLights();

	void LoadCornellBox();
	void LoadCornellBoxSmall();
	void LoadSibernik();
	void LoadSimpleScene();
	void LoadCornellBoxDragon();
	void LoadBuddha();
	void LoadCornellEmpty();
	void LoadRoom();
	void LoadConferenceRoom();
	void LoadHouse();

	bool HasLightSource() { return m_HasLightSource; }

	bool IntersectRayScene(const Ray& ray, float* t, Intersection *pIntersection, CPrimitive::IsectMode isectMode);
	bool IntersectRaySceneSimple(const Ray& ray, float* t, Intersection *pIntersection, CPrimitive::IsectMode isectMode);

	CCamera* GetCamera() { return m_Camera; }

	void ClearLighting();
	void SetAVPLImportanceSampling(CAVPLImportanceSampling *pAVPLIS) { m_pAVPLImportanceSampling = pAVPLIS; }

	int GetNumberOfLightPaths() { return m_NumLightPaths; }
				
	void CreatePaths(std::vector<AVPL>& avpls, std::vector<AVPL>& allAVPLs, std::vector<AVPL>& isAVPLs, bool profile, uint numPaths);
	void CreatePath(std::vector<AVPL>& avpls);
	void CreatePrimaryVpls(std::vector<AVPL>& avpls, int numVpls);
	bool CreateAVPL(AVPL* pred, AVPL* newAVPL);
	bool CreatePrimaryAVPL(AVPL* newAVPL);

	uint GetNumCreatedAVPLs() { return m_NumCreatedAVPLs; }
	uint GetNumAVPLsAfterIS() { return m_NumAVPLsAfterIS; }

	bool ImportanceSampling(AVPL& avpl, float* scale);
	bool Visible(const SceneSample& ss1, const SceneSample& ss2);
	void SampleLightSource(SceneSample& ss);

	CMaterialBuffer* GetMaterialBuffer() { return m_pMaterialBuffer; }
	MATERIAL* GetMaterial(const SceneSample& ss);

	CReferenceImage* GetReferenceImage() { return m_pReferenceImage; }

private:
	void ClearPath();
		
	void InitKdTree();
	void ReleaseKdTree();

	
	bool ContinueAVPLPath(AVPL* pred, AVPL* newAVPL, glm::vec3 direction, float pdf);
	void CreateAVPLs(AVPL* pred, std::vector<AVPL>& path, int nAVPLs);
	
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

	bool m_HasLightSource;

	CMaterialBuffer* m_pMaterialBuffer;
	CReferenceImage* m_pReferenceImage;
};

#endif SCENE_H