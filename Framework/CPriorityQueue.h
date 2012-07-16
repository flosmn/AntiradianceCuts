#ifndef _C_PRIORITY_QUEUE_H_
#define _C_PRIORITY_QUEUE_H_

#include "LightTreeTypes.h"

class CTimer;

namespace PriorityQueue
{

struct ELEMENT{
	ELEMENT(int i, float v) : index(i), value(v) {}
	ELEMENT() {} 

	int index;
	float value;
};

class CPriorityQueue
{
public:
	CPriorityQueue(int size);
	CPriorityQueue();
    ~CPriorityQueue();

	void Release();

    void Insert(const CLUSTER_PAIR& cp);
    CLUSTER_PAIR DeleteMin();
	bool Empty() { return m_numElements == 0; }

	void PrintTimes();

private:
	int Left(int parent);
    int Right(int parent);
    int Parent(int child);
    void HeapifyUp(int index);
    void HeapifyDown(int index);
	inline void SWAP(int index1, int index2);
	void IncreaseHeapSize();

	CLUSTER_PAIR* m_pClusterPairsMap;
	int m_ClusterPairMapSize;
	int m_ClusterPairMapPointer;
	int* m_pIndexUsed;
	
	ELEMENT* m_pElements;
	int m_numElements;
	int m_Size;

	double m_InsertTime;
	double m_InsertHeapifyTime;
	double m_InsertIncreaseTime;
	double m_InsertRestTime;
	double m_InsertMap;
	double m_DeleteTime;
	CTimer* m_pTimer;
	CTimer* m_pTimer2;
};

};

#endif // _C_PRIORITY_QUEUE_H_