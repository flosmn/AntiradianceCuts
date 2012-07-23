#include "LightTreeTypes.h"

bool SORT_X(CLUSTER* p1, CLUSTER* p2)
{
	return (p1->mean.x < p2->mean.x);
}

bool SORT_Y(CLUSTER* p1, CLUSTER* p2)
{
	return (p1->mean.y < p2->mean.y);
}

bool SORT_Z(CLUSTER* p1, CLUSTER* p2)
{
	return (p1->mean.z < p2->mean.z);
}