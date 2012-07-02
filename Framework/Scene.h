#ifndef SCENE_H
#define SCENE_H

typedef unsigned int uint;

#include "GL/glew.h"

#include "Ray.h"
#include "Intersection.h"

#include "Utils\mtrand.h"

#include <vector>

class Camera;
class AVPL;
class AreaLight;
class CKdTreeAccelerator;
class CPbrtKdTreeAccel;

class COGLUniformBuffer;

class CModel;

class Scene
{
public:
	Scene(Camera* pCamera);
	~Scene();
	
	bool Init();
	void Release();
	
	void ClearScene();
	
	void DrawScene(COGLUniformBuffer* pUBTransform, COGLUniformBuffer* pUBMaterial);
	void DrawScene(COGLUniformBuffer* pUBTransform);

	void DrawScene(const glm::mat4& mView, const glm::mat4& mProj, COGLUniformBuffer* pUBTransform);

	void DrawAreaLight(COGLUniformBuffer* pUBTransform, COGLUniformBuffer* pUBAreaLight);

	void LoadCornellBox();
	void LoadSimpleScene();
		
	Camera* GetCamera() { return m_Camera; }

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

	Camera* m_Camera;		
	AreaLight* m_AreaLight;

	CKdTreeAccelerator* m_pKdTreeAccelerator;
	CPbrtKdTreeAccel* m_pPbrtKdTreeAccelerator;
	std::vector<CPrimitive*> m_Primitives;

};

#endif SCENE_H