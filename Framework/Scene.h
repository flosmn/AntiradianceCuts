#ifndef SCENE_H
#define SCENE_H

typedef unsigned int uint;

#include "GL/glew.h"

#include "Ray.h"
#include "Intersection.h"

#include "CUtils\mtrand.h"

#include <vector>

class Camera;
class Light;
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

	void DrawAreaLight(CGLUniformBuffer* pUBTransform);

	void LoadCornellBox();
	void LoadSimpleScene();
		
	Camera* GetCamera() { return m_Camera; }

	void ClearLighting();

	int GetNumberOfLightPaths() { return m_NumLightPaths; }

	bool IntersectRayScene(Ray ray, Intersection &intersection);
		
	std::vector<Light*> CreatePaths(uint numPaths, int N, int nAdditionalAVPLs, bool useHammersley);
	std::vector<Light*> CreatePath(int N, int nAdditionalAVPLs, bool useHammersley);
	
	void Stats();

	void CreatePlaneHammersleySamples(int i);
		
private:
	void ClearPath();

	Light* CreateLight(Light* tail, int N, int nAdditionalAVPLs, bool useHammersley);
	Light* CreateLight(Light* tail, glm::vec3 direction, float pdf, int N, int nAdditionalAVPLs, bool useHammersley);
	void CreateAVPLs(Light* tail, std::vector<Light*>& path, int N, int nAVPLs, bool useHammersley);

	std::vector<CModel*> m_Models;
		
	float* m_pPlaneHammersleySamples;

	int m_CurrentBounce;
	int m_NumLightPaths;

	Camera* m_Camera;		
	AreaLight* m_AreaLight;

	int *m_BounceInfo;
	int m_MaxBounceInfo;
	glm::vec3 m_MaxVPLFlow;
	int m_MaxVPLFlowBounce;
	glm::vec3 m_Alpha;
};

#endif SCENE_H