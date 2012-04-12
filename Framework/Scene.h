#ifndef SCENE_H
#define SCENE_H

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
	
	void SetDebugColor(Light* light, int bounce);

	std::vector<Light*> CreatePathPBRT();
	std::vector<Light*> CreatePath();
	std::vector<Light*> GetCurrentPath() { return m_CurrentPath; } 	
	std::vector<Light*> GetPath(int i) { return m_Paths[i]; } 	
	std::vector<Light*> GetLights() { return m_Lights; }

	void Stats();
		
private:
	void ClearPath();

	Light* CreateLight(Light* tail);

	
	std::vector<CModel*> m_Models;
	std::vector<Light*> m_Lights;
	std::vector<std::vector<Light*>> m_Paths;
	std::vector<Light*> m_CurrentPath;

	glm::vec3 m_AvgReflectivity;
	float m_MeanRho;
	
	Camera* m_Camera;		
	AreaLight* m_AreaLight;

	int *m_BounceInfo;
	int m_MaxBounceInfo;
	glm::vec3 m_MaxVPLFlow;
	int m_MaxVPLFlowBounce;

	int m_CurrentBounce;	
	glm::vec3 m_Alpha;
};

#endif SCENE_H