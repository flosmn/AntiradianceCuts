#include "CTestFramework.h"

#include <glm/glm.hpp>

#include "Triangle.h"
#include "KdTreeAccelerator.h"
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

	Triangle triangle(p0, p1, p2);

	glm::vec3 min = glm::min(glm::min(p0, p1), p2);
	glm::vec3 max = glm::max(glm::max(p0, p1), p2);

	BBox bbox = triangle.getBBox();

	assert(bbox.getMin() == min);
	assert(bbox.getMax() == max);

	Ray ray0(glm::vec3(-0.5f, 0.5f, 1.f), glm::vec3(0.f, 0.f, -1.f));
	Ray ray1(glm::vec3(-0.0f, 0.0f, 1.f), glm::vec3(0.f, 0.f, -1.f));
	Ray ray2(glm::vec3(-0.5f, -0.5f, 1.25f), glm::vec3(0.f, 0.f, -1.f));
	Ray ray3(glm::vec3(-1.5f, -0.f, 1.f), glm::vec3(0.f, 0.f, -1.f));

	float t = -10000.f;

	Intersection intersection;
	assert(triangle.intersectBBox(ray0) == true);
	assert(triangle.intersect(ray0, &t, &intersection, Triangle::FRONT_FACE) == false);

	assert(triangle.intersectBBox(ray1) == true);
	assert(triangle.intersect(ray1, &t, &intersection, Triangle::FRONT_FACE) == true);
	assert(t == 1.f);

	assert(triangle.intersectBBox(ray2) == true);
	assert(triangle.intersect(ray2, &t, &intersection, Triangle::FRONT_FACE) == true);
	assert(t == 1.25f);

	assert(triangle.intersectBBox(ray3) == false);
	assert(triangle.intersect(ray3, &t, &intersection, Triangle::FRONT_FACE) == false);
}

void CTestFramework::KdTreeBuildTest()
{
	std::vector<Triangle> primitives;
	primitives.push_back(Triangle(
		glm::vec3(0.1f, 0.6f, 0.0f),
		glm::vec3(0.4f, 0.6f, 0.0f),
		glm::vec3(0.25f, 0.9f, 0.0f)));
	primitives.push_back(Triangle(
		glm::vec3(0.6f, 0.75f, 0.0f),
		glm::vec3(0.9f, 0.5f, 0.0f),
		glm::vec3(0.9f, 0.9f, 0.0f)));
	primitives.push_back(Triangle(
		glm::vec3(0.1f, 0.25f, 0.0f),
		glm::vec3(0.4f, 0.1f, 0.0f),
		glm::vec3(0.4f, 0.4f, 0.0f)));
	primitives.push_back(Triangle(
		glm::vec3(0.75f, 0.1f, 0.0f),
		glm::vec3(0.9f, 0.4f, 0.0f),
		glm::vec3(0.6f, 0.4f, 0.0f)));
	
	primitives.push_back(Triangle(
		glm::vec3(0.1f, 0.6f, 0.5f),
		glm::vec3(0.4f, 0.6f, 0.5f),
		glm::vec3(0.25f, 0.9f, 0.5f)));
	primitives.push_back(Triangle(
		glm::vec3(0.1f, 0.25f, 0.5f),
		glm::vec3(0.4f, 0.1f, 0.5f),
		glm::vec3(0.4f, 0.4f, 0.5f)));
	
	const int MAX_NUM_NODES = 1;

	KdTreeAccelerator kdTree(primitives, 80, 1, MAX_NUM_NODES, 20);
	kdTree.buildTree();

	kdTree.printForDebug();

	int numNodes = kdTree.getNumNodes();
	KdAccelNode* nodes = kdTree.getNodes();
	std::vector<Triangle> primitivesInTree;
	for(int i = 0; i < numNodes; ++i)
	{
		if(nodes[i].isLeaf())
		{
			std::cout << "Leaf has " << nodes[i].getNumPrimitives() << " primitives." << std::endl;
			std::vector<Triangle> nodePrimitives = kdTree.getPrimitivesOfNode(i);
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
	assert(kdTree.intersect(ray0, &t, &intersection, Triangle::BACK_FACE) == true);
	assert(intersection.getPosition() == glm::vec3(0.25f, 0.75f, 0.5f));
	assert(t = 0.5f);

	assert(kdTree.intersect(ray1, &t, &intersection, Triangle::BACK_FACE) == true);
	assert(t = 0.75f);

	assert(kdTree.intersect(ray2, &t, &intersection, Triangle::BACK_FACE) == true);
	assert(t = 1.f);

	assert(kdTree.intersect(ray3, &t, &intersection, Triangle::BACK_FACE) == true);
	assert(t = 1.f);

	assert(kdTree.intersect(ray4, &t, &intersection, Triangle::BACK_FACE) == false);
	assert(kdTree.intersect(ray5, &t, &intersection, Triangle::BACK_FACE) == false);
}