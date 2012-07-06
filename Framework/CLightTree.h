#ifndef C_LIGHT_TREE_H_
#define C_LIGHT_TREE_H_

class AVPL;

#include <glm/glm.hpp>

#include "BBox.h"

#include <vector>

typedef unsigned int uint;

struct CLUSTER;

class CLightTree
{
public:
	CLightTree();
	~CLightTree();

	void BuildTree(const std::vector<AVPL*>& avpls, const float weightNormals);
	void BuildTreeTweakCP(const std::vector<AVPL*>& avpls, const float weightNormals);
	void Color(const std::vector<AVPL*>& avpls, const int cutDepth);

	void Release();

	CLUSTER* GetHead();

private:

	void Traverse(CLUSTER* cluster);
	void Release(CLUSTER* cluster);
	void Color(const std::vector<AVPL*>& avpls, const int cutDepth, CLUSTER* cluster, const int currentDepth, const int colorIndex);
	void GetAllLeafs(CLUSTER* cluster, std::vector<CLUSTER*>& leafs);

	CLUSTER* m_Head;

	glm::vec3 GetRandomColor();
	void InitColors();

	glm::vec3* m_pColors;
	int m_NumColors;	
};

#endif C_LIGHT_TREE_H_