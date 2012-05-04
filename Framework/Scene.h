#ifndef SCENE_H
#define SCENE_H

typedef unsigned int uint;

#include "GL/glew.h"

#include "Ray.h"
#include "Intersection.h"

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

	int GetNumberOfLightPaths() { return m_Paths.size(); }

	bool IntersectRayScene(Ray ray, Intersection &intersection);
		
	std::vector<Light*> CreatePathPBRT();
	std::vector<Light*> CreatePaths(uint numPaths, uint currentNumberOfPaths, int N, int nAdditionalAVPLs, bool useHammersley);
	std::vector<Light*> CreatePath(uint currentNumberOfPaths, int N, int nAdditionalAVPLs, bool useHammersley);
	std::vector<Light*> GetPath(int i) { return m_Paths[i]; } 	
	
	void Stats();

	void CreatePlaneHammersleySamples(int i);
		
private:
	void ClearPath();

	Light* CreateLight(Light* tail, int N, int nAdditionalAVPLs, bool useHammersley);
	
	std::vector<CModel*> m_Models;
	std::vector<std::vector<Light*>> m_Paths;
	
	float* m_pPlaneHammersleySamples;

	int m_CurrentBounce;	

	Camera* m_Camera;		
	AreaLight* m_AreaLight;

	int *m_BounceInfo;
	int m_MaxBounceInfo;
	glm::vec3 m_MaxVPLFlow;
	int m_MaxVPLFlowBounce;
	glm::vec3 m_Alpha;
};

#endif SCENE_H