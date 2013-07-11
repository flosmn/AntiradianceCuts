#ifndef SCENE_H
#define SCENE_H

typedef unsigned int uint;

#include "GL/glew.h"

#include "Ray.h"
#include "Intersection.h"
#include "SceneSample.h"
#include "Triangle.h"
#include "Material.h"

#include "Utils\mtrand.h"

#include <vector>
#include <memory>

class CCamera;
class AVPL;
class AreaLight;
class KdTreeAccelerator;
class CConfigManager;
class CAVPLImportanceSampling;
class CMaterialBuffer;
class CReferenceImage;

class COGLUniformBuffer;
class COCLContext;

class Model;
class Mesh;

class Scene
{
public:
	Scene(CCamera* camera, CConfigManager* confManager, COCLContext* clContext);
	~Scene();
	
	void ClearScene();
	
	void UpdateAreaLights();

	void LoadCornellBox();
	void LoadSibernik();
	void LoadSimpleScene();
	void LoadBuddha();
	void LoadCornellEmpty();
	void LoadConferenceRoom();

	bool HasLightSource() { return m_HasLightSource; }

	bool IntersectRayScene(const Ray& ray, float* t, Intersection *pIntersection, Triangle::IsectMode isectMode);
	
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

	BBox getBoundingBox() { return m_bbox; }
	float getSceneExtent() { return glm::length(m_bbox.getMax() - m_bbox.getMin()); }

	std::vector<std::unique_ptr<Model>> const&  getModels() const { return m_models; }
	AreaLight const* getAreaLight() const { return m_areaLight.get(); }

private:
	void loadSceneFromFile(std::string const& file);
	void calcBoundingBox();
	void ClearPath();
		
	void initKdTree();
	void ReleaseKdTree();
	
	bool ContinueAVPLPath(AVPL* pred, AVPL* newAVPL, glm::vec3 direction, float pdf);
	void CreateAVPLs(AVPL* pred, std::vector<AVPL>& path, int nAVPLs);
	
private:
	BBox m_bbox;

	CCamera* m_camera;		
	CConfigManager* m_confManager;

	std::unique_ptr<AreaLight> m_areaLight;
	std::unique_ptr<KdTreeAccelerator> m_kdTreeAccelerator;

	std::unique_ptr<CMaterialBuffer> m_materialBuffer;
	std::unique_ptr<CReferenceImage> m_referenceImage;

	std::vector<std::unique_ptr<Mesh>> m_meshes;
	std::vector<std::unique_ptr<Model>> m_models;

	uint m_NumCreatedAVPLs;

	int m_CurrentBounce;
	int m_NumLightPaths;

	bool m_HasLightSource;
};

#endif SCENE_H
