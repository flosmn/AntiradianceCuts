#include "CTestFramework.h"

#include <glm/glm.hpp>

#include "CTriangle.h"
#include "CKdTreeAccelerator.h"
#include "Ray.h"
#include "BBox.h"

#include <assert.h>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

CTestFramework::CTestFramework()
{
}

CTestFramework::~CTestFramework()
{
}

void CTestFramework::RunTests()
{
	TriangleIntersectionTest();
	KdTreeBuildTest();
}

void CTestFramework::TriangleIntersectionTest()
{
	glm::vec3 p0(-1.f, -1.f, 0.f);
	glm::vec3 p1( 1.f, -1.f, 0.f);
	glm::vec3 p2( 0.f,  1.f, 0.f);

	CTriangle triangle(p0, p1, p2);

	glm::vec3 min = glm::min(glm::min(p0, p1), p2);
	glm::vec3 max = glm::max(glm::max(p0, p1), p2);

	BBox bbox = triangle.GetBBox();

	assert(bbox.pMin == min);
	assert(bbox.pMax == max);

	Ray ray0(glm::vec3(-0.5f, 0.5f, 1.f), glm::vec3(0.f, 0.f, -1.f));
	Ray ray1(glm::vec3(-0.0f, 0.0f, 1.f), glm::vec3(0.f, 0.f, -1.f));
	Ray ray2(glm::vec3(-0.5f, -0.5f, 1.25f), glm::vec3(0.f, 0.f, -1.f));
	Ray ray3(glm::vec3(-1.5f, -0.f, 1.f), glm::vec3(0.f, 0.f, -1.f));

	float t = -10000.f;

	Intersection intersection;
	assert(triangle.IntersectBBox(ray0) == true);
	assert(triangle.Intersect(ray0, &t, &intersection, CTriangle::FRONT_FACE) == false);

	assert(triangle.IntersectBBox(ray1) == true);
	assert(triangle.Intersect(ray1, &t, &intersection, CTriangle::FRONT_FACE) == true);
	assert(t == 1.f);

	assert(triangle.IntersectBBox(ray2) == true);
	assert(triangle.Intersect(ray2, &t, &intersection, CTriangle::FRONT_FACE) == true);
	assert(t == 1.25f);

	assert(triangle.IntersectBBox(ray3) == false);
	assert(triangle.Intersect(ray3, &t, &intersection, CTriangle::FRONT_FACE) == false);
}

void CTestFramework::KdTreeBuildTest()
{
	std::vector<CTriangle> primitives;
	primitives.push_back(CTriangle(
		glm::vec3(0.1f, 0.6f, 0.0f),
		glm::vec3(0.4f, 0.6f, 0.0f),
		glm::vec3(0.25f, 0.9f, 0.0f)));
	primitives.push_back(CTriangle(
		glm::vec3(0.6f, 0.75f, 0.0f),
		glm::vec3(0.9f, 0.5f, 0.0f),
		glm::vec3(0.9f, 0.9f, 0.0f)));
	primitives.push_back(CTriangle(
		glm::vec3(0.1f, 0.25f, 0.0f),
		glm::vec3(0.4f, 0.1f, 0.0f),
		glm::vec3(0.4f, 0.4f, 0.0f)));
	primitives.push_back(CTriangle(
		glm::vec3(0.75f, 0.1f, 0.0f),
		glm::vec3(0.9f, 0.4f, 0.0f),
		glm::vec3(0.6f, 0.4f, 0.0f)));
	
	primitives.push_back(CTriangle(
		glm::vec3(0.1f, 0.6f, 0.5f),
		glm::vec3(0.4f, 0.6f, 0.5f),
		glm::vec3(0.25f, 0.9f, 0.5f)));
	primitives.push_back(CTriangle(
		glm::vec3(0.1f, 0.25f, 0.5f),
		glm::vec3(0.4f, 0.1f, 0.5f),
		glm::vec3(0.4f, 0.4f, 0.5f)));
	
	const int MAX_NUM_NODES = 1;

	CKdTreeAccelerator kdTree(primitives, 80, 1, MAX_NUM_NODES, 20);
	kdTree.BuildTree();

	kdTree.PrintForDebug();

	int numNodes = kdTree.GetNumNodes();
	KdAccelNode* nodes = kdTree.GetNodes();
	std::vector<CTriangle> primitivesInTree;
	for(int i = 0; i < numNodes; ++i)
	{
		if(nodes[i].IsLeaf())
		{
			std::cout << "Leaf has " << nodes[i].GetNumPrimitives() << " primitives." << std::endl;
			std::vector<CTriangle> nodePrimitives = kdTree.GetPrimitivesOfNode(i);
			primitivesInTree.insert(primitivesInTree.end(), nodePrimitives.begin(), nodePrimitives.end());
		}
	}
	for(uint i = 0; i < primitives.size(); ++i)
	{
		//assert(std::find(primitivesInTree.begin(), primitivesInTree.end(), primitives[i])) != primitivesInTree.end()));
	}

	Ray ray0(glm::vec3(0.25f, 0.75f, 1.f),		glm::vec3(0.f, 0.f, -1.f));
	Ray ray1(glm::vec3(0.25f, 0.75f, 1.25f),	glm::vec3(0.f, 0.f, -1.f));
	Ray ray2(glm::vec3(0.25f, 0.25f, 1.0f),		glm::vec3(0.f, 0.f, -1.f));
	Ray ray3(glm::vec3(0.75f, 0.25f, 1.0f),		glm::vec3(0.f, 0.f, -1.f));
	Ray ray4(glm::vec3(0.5f, 0.5f, 1.0f),		glm::vec3(0.f, 0.f, -1.f));
	Ray ray5(glm::vec3(1.25f, 0.5f, 1.0f),		glm::vec3(0.f, 0.f, -1.f));

	float t = 1000000.f;

	Intersection intersection;
	assert(kdTree.Intersect(ray0, &t, &intersection, CTriangle::FRONT_FACE) == true);
	assert(intersection.GetPosition() == glm::vec3(0.25f, 0.75f, 0.5f));
	assert(t = 0.5f);

	assert(kdTree.Intersect(ray1, &t, &intersection, CTriangle::FRONT_FACE) == true);
	assert(t = 0.75f);

	assert(kdTree.Intersect(ray2, &t, &intersection, CTriangle::FRONT_FACE) == true);
	assert(t = 1.f);

	assert(kdTree.Intersect(ray3, &t, &intersection, CTriangle::FRONT_FACE) == true);
	assert(t = 1.f);

	assert(kdTree.Intersect(ray4, &t, &intersection, CTriangle::FRONT_FACE) == false);
	assert(kdTree.Intersect(ray5, &t, &intersection, CTriangle::FRONT_FACE) == false);
}