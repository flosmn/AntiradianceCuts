#include "Rand.h"

#define _CRT_RAND_S
#include <stdlib.h>

#include <omp.h>

#include <iostream>

void RandInit(int i)
{
	//mt_rand = new MTRand(i);
}

void RandSeed(int i)
{
	//mt_rand->seed(i);
}

float Rand01()
{
	//return (float)(*mt_rand)();

	unsigned int number;
    errno_t err = rand_s( &number );
    return (float) number / (float) UINT_MAX;
}