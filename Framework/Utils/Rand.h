#ifndef _RAND_H_
#define _RAND_H_

#include "mtrand.h"

static MTRand* mt_rand;

void RandInit(int i);
void RandSeed(int i);

float Rand01();

#endif _RAND_H_