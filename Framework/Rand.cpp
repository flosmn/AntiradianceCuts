#include "Rand.h"

void RandInit(int i)
{
	mt_rand = new MTRand(i);
}

void RandSeed(int i)
{
	mt_rand->seed(i);
}

float Rand01()
{
	return (float)(*mt_rand)();
}