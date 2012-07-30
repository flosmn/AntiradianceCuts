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
	void LoadSibernik();
	void LoadSimpleScene();
	void LoadCornellBoxDragon();
		
	CCamera* GetCamera() { return m_Camera; }

	void ClearLighting();

	int GetNumberOfLightPaths() { return m_NumLightPaths; }
				
	std::vector<AVPL*> CreatePaths(uint numPaths, int N, int nAdditionalAVPLs);
	std::vector<AVPL*> CreatePath(int N, int nAdditionalAVPLs);
	std::vector<AVPL*> CreatePrimaryVpls(int numVpls);

private:
	void ClearPath();

	bool IntersectRaySceneSimple(const Ray& ray, float* t, Intersection *pIntersection);

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

	CConfigManager* m_pConfManager;
};

#endif SCENE_H