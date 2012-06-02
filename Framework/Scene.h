#ifndef SCENE_H
#define SCENE_H

typedef unsigned int uint;

#include "GL/glew.h"

#include "Ray.h"
#include "Intersection.h"

#include "CUtils\mtrand.h"

#include <vector>

class Camera;
class AVPL;
class AreaLight;

class CGLUniformBuffer;

class CModel;

class Scene
{
public:
	Scene(Camera* pCamera);
	~Scene();
	
	bool Init();
	void Release();

	void ClearScene();
	
	void DrawScene(CGLUniformBuffer* pUBTransform, CGLUniformBuffer* pUBMaterial);
	void DrawScene(CGLUniformBuffer* pUBTransform);

	void DrawScene(const glm::mat4& mView, const glm::mat4& mProj, CGLUniformBuffer* pUBTransform);

	void DrawAreaLight(CGLUniformBuffer* pUBTransform, CGLUniformBuffer* pUBAreaLight);

	void LoadCornellBox();
	void LoadSimpleScene();
		
	Camera* GetCamera() { return m_Camera; }

	void ClearLighting();

	int GetNumberOfLightPaths() { return m_NumLightPaths; }

	bool IntersectRayScene(Ray ray, Intersection &intersection);
		
	std::vector<AVPL*> CreatePaths(uint numPaths, int N, int nAdditionalAVPLs, bool useHammersley);
	std::vector<AVPL*> CreatePath(int N, int nAdditionalAVPLs, bool useHammersley);
	
	void CreatePlaneHammersleySamples(int i);
		
private:
	void ClearPath();

	AVPL* CreateAVPL(AVPL* pred, int N, int nAdditionalAVPLs, bool useHammersley);
	AVPL* ContinueAVPLPath(AVPL* pred, glm::vec3 direction, float pdf, int N, int nAdditionalAVPLs, bool useHammersley);
	void CreateAVPLs(AVPL* pred, std::vector<AVPL*>& path, int N, int nAVPLs, bool useHammersley);

	std::vector<CModel*> m_Models;
		
	float* m_pPlaneHammersleySamples;

	int m_CurrentBounce;
	int m_NumLightPaths;

	Camera* m_Camera;		
	AreaLight* m_AreaLight;
};

#endif SCENE_H