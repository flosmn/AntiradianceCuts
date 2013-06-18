#ifndef SCENE_H
#define SCENE_H

typedef unsigned int uint;

#include "GL/glew.h"

#include "Ray.h"
#include "Intersection.h"
#include "CTriangle.h"
#include "Material.h"

#include "Utils\mtrand.h"

#include <vector>
#include <memory>

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
	Scene(CCamera* camera, CConfigManager* confManager, COCLContext* clContext);
	~Scene();
	
	void ClearScene();
	
	void DrawScene(COGLUniformBuffer* ubTransform, COGLUniformBuffer* ubMaterial);
	void DrawScene(COGLUniformBuffer* ubTransform);

	void DrawScene(const glm::mat4& mView, const glm::mat4& mProj, COGLUniformBuffer* ubTransform);

	void DrawAreaLight(COGLUniformBuffer* ubTransform, COGLUniformBuffer* ubAreaLight);
	void DrawAreaLight(COGLUniformBuffer* ubTransform, COGLUniformBuffer* ubAreaLight, glm::vec3 color);

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

	bool IntersectRayScene(const Ray& ray, float* t, Intersection *pIntersection, CTriangle::IsectMode isectMode);
	bool IntersectRaySceneSimple(const Ray& ray, float* t, Intersection *pIntersection, CTriangle::IsectMode isectMode);

	CCamera* GetCamera() { return m_camera; }

	void ClearLighting();

	int GetNumberOfLightPaths() { return m_NumLightPaths; }
				
	void CreatePaths(std::vector<AVPL>& avpls, std::vector<AVPL>& allAVPLs, std::vector<AVPL>& isAVPLs, bool profile, uint numPaths);
	void CreatePath(std::vector<AVPL>& avpls);
	void CreatePrimaryVpls(std::vector<AVPL>& avpls, int numVpls);
	bool CreateAVPL(AVPL* pred, AVPL* newAVPL);
	bool CreatePrimaryAVPL(AVPL* newAVPL);

	uint GetNumCreatedAVPLs() { return m_NumCreatedAVPLs; }

	CMaterialBuffer* GetMaterialBuffer() { return m_materialBuffer.get(); }
	bool Visible(const SceneSample& ss1, const SceneSample& ss2);
	void SampleLightSource(SceneSample& ss);	MATERIAL* GetMaterial(const SceneSample& ss);

	CReferenceImage* GetReferenceImage() { return m_referenceImage.get(); }

private:
	void ClearPath();
		
	void InitKdTree();
	void ReleaseKdTree();
	
	bool ContinueAVPLPath(AVPL* pred, AVPL* newAVPL, glm::vec3 direction, float pdf);
	void CreateAVPLs(AVPL* pred, std::vector<AVPL>& path, int nAVPLs);
	
private:
	std::vector<std::unique_ptr<CModel>> m_models;
	std::vector<CTriangle> m_primitives;

	CCamera* m_camera;		
	CConfigManager* m_confManager;

	std::unique_ptr<AreaLight> m_areaLight;
	std::unique_ptr<CKdTreeAccelerator> m_kdTreeAccelerator;

	std::unique_ptr<CMaterialBuffer> m_materialBuffer;
	std::unique_ptr<CReferenceImage> m_referenceImage;

	uint m_NumCreatedAVPLs;

	int m_CurrentBounce;
	int m_NumLightPaths;

	bool m_HasLightSource;
};

#endif SCENE_H
