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

	IntersectionNew intersection;
	assert(triangle.IntersectBBox(ray0) == true);
	assert(triangle.Intersect(ray0, &t, &intersection) == false);

	assert(triangle.IntersectBBox(ray1) == true);
	assert(triangle.Intersect(ray1, &t, &intersection) == true);
	assert(t == 1.f);

	assert(triangle.IntersectBBox(ray2) == true);
	assert(triangle.Intersect(ray2, &t, &intersection) == true);
	assert(t == 1.25f);

	assert(triangle.IntersectBBox(ray3) == false);
	assert(triangle.Intersect(ray3, &t, &intersection) == false);
}

void CTestFramework::KdTreeBuildTest()
{
	std::vector<CPrimitive*> primitives;
	primitives.push_back(new CTriangle(
		glm::vec3(0.1f, 0.6f, 0.0f),
		glm::vec3(0.4f, 0.6f, 0.0f),
		glm::vec3(0.25f, 0.9f, 0.0f)));
	primitives.push_back(new CTriangle(
		glm::vec3(0.6f, 0.75f, 0.0f),
		glm::vec3(0.9f, 0.5f, 0.0f),
		glm::vec3(0.9f, 0.9f, 0.0f)));
	primitives.push_back(new CTriangle(
		glm::vec3(0.1f, 0.25f, 0.0f),
		glm::vec3(0.4f, 0.1f, 0.0f),
		glm::vec3(0.4f, 0.4f, 0.0f)));
	primitives.push_back(new CTriangle(
		glm::vec3(0.75f, 0.1f, 0.0f),
		glm::vec3(0.9f, 0.4f, 0.0f),
		glm::vec3(0.6f, 0.4f, 0.0f)));
	primitives.push_back(new CTriangle(
		glm::vec3(0.1f, 0.6f, 0.5f),
		glm::vec3(0.4f, 0.6f, 0.5f),
		glm::vec3(0.25f, 0.9f, 0.5f)));
	primitives.push_back(new CTriangle(
		glm::vec3(0.1f, 0.25f, 0.5f),
		glm::vec3(0.4f, 0.1f, 0.5f),
		glm::vec3(0.4f, 0.4f, 0.5f)));
	
	CKdTreeAccelerator kdTree(primitives,
		80, 1, 1, 3);
	kdTree.BuildTree();

	Ray ray0(glm::vec3(0.25f, 0.75f, 1.f),		glm::vec3(0.f, 0.f, -1.f));
	Ray ray1(glm::vec3(0.25f, 0.75f, 1.25f),	glm::vec3(0.f, 0.f, -1.f));
	Ray ray2(glm::vec3(0.25f, 0.25f, 1.0f),		glm::vec3(0.f, 0.f, -1.f));
	Ray ray3(glm::vec3(0.75f, 0.25f, 1.0f),		glm::vec3(0.f, 0.f, -1.f));
	Ray ray4(glm::vec3(0.5f, 0.5f, 1.0f),		glm::vec3(0.f, 0.f, -1.f));
	Ray ray5(glm::vec3(1.25f, 0.5f, 1.0f),		glm::vec3(0.f, 0.f, -1.f));

	float t = 1000000.f;

	IntersectionNew intersection;
	assert(kdTree.Intersect(ray0, &t, &intersection) == true);
	assert(t = 0.5f);

	assert(kdTree.Intersect(ray1, &t, &intersection) == true);
	assert(t = 0.75f);

	assert(kdTree.Intersect(ray2, &t, &intersection) == true);
	assert(t = 1.f);

	assert(kdTree.Intersect(ray3, &t, &intersection) == true);
	assert(t = 1.f);

	assert(kdTree.Intersect(ray4, &t, &intersection) == false);
	assert(kdTree.Intersect(ray5, &t, &intersection) == false);
}