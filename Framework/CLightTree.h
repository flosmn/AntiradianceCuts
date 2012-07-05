#ifndef C_LIGHT_TREE_H_
#define C_LIGHT_TREE_H_

class AVPL;

#include <glm/glm.hpp>

#include "BBox.h"

#include <vector>

typedef unsigned int uint;

struct CLUSTER
{
	CLUSTER* left;
	CLUSTER* right;

	uint size;
	BBox bbox;
	glm::vec3 intensity;
	glm::vec3 normal;
};

class CLightTree
{
public:
	CLightTree();
	~CLightTree();

	void BuildTree(const std::vector<AVPL*>& avpls);

private:
	
};

#endif C_LIGHT_TREE_H_