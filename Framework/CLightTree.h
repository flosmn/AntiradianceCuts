#ifndef C_LIGHT_TREE_H_
#define C_LIGHT_TREE_H_

class AVPL;

#include <glm/glm.hpp>

#include "BBox.h"

#include <vector>

typedef unsigned int uint;

class CLightTree
{
public:
	CLightTree();
	~CLightTree();

	void BuildTree(const std::vector<AVPL*>& avpls, const float& weightNormals);

private:
	
};

#endif C_LIGHT_TREE_H_