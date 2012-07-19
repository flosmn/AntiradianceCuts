#include "CPriorityQueue.h"

#include <iostream>

using namespace PriorityQueue;

CPriorityQueue::CPriorityQueue(int size)
{
	m_Size = size;
	m_numElements = 0;
	m_pElements = new ELEMENT[m_Size];
	
	m_ClusterPairMapSize = 2 * m_Size;
	m_ClusterPairMapPointer = 0;
	m_pClusterPairsMap = new CLUSTER_PAIR[m_ClusterPairMapSize];
	
	m_pIndexUsed = new int[m_ClusterPairMapSize];
	memset(m_pIndexUsed, 0, m_ClusterPairMapSize * sizeof(int));	
}

CPriorityQueue::~CPriorityQueue()
{
	if(m_pElements)
		delete [] m_pElements;
	if(m_pClusterPairsMap)
		delete [] m_pClusterPairsMap;
	if(m_pIndexUsed)
		delete [] m_pIndexUsed;
}

void CPriorityQueue::Release()
{
	if(m_pIndexUsed)
		memset(m_pIndexUsed, 0, m_ClusterPairMapSize * sizeof(int));
	
	m_ClusterPairMapPointer = 0;
	m_numElements = 0;
	
	m_pElements = new ELEMENT[m_Size];
}

void CPriorityQueue::Insert(const CLUSTER_PAIR& cp)
{	
	if(m_numElements >= m_Size)
		IncreaseHeapSize();
		
	while(m_pIndexUsed[m_ClusterPairMapPointer] != 0)
		m_ClusterPairMapPointer = (m_ClusterPairMapPointer + 1) % m_ClusterPairMapSize;

	ELEMENT e(m_ClusterPairMapPointer, cp.dist);
	m_pClusterPairsMap[m_ClusterPairMapPointer] = cp;
	m_pIndexUsed[m_ClusterPairMapPointer] = 1;

	m_pElements[m_numElements] = e;
    
	m_numElements++;
	
	HeapifyUp(m_numElements - 1);
}

CLUSTER_PAIR CPriorityQueue::DeleteMin()
{
	ELEMENT min = m_pElements[0];
	CLUSTER_PAIR cp = m_pClusterPairsMap[min.index];
	m_pIndexUsed[min.index] = 0;
	
	m_numElements--;

	m_pElements[0] = m_pElements[m_numElements];
	
	HeapifyDown(0);
	
	return cp;
}

int CPriorityQueue::Left(int parent)
{
	int i = ( parent << 1 ) + 1; // 2 * parent + 1
    return ( i < m_numElements ) ? i : -1;
}

int CPriorityQueue::Right(int parent)
{
	int i = ( parent << 1 ) + 2; // 2 * parent + 2
    return ( i < m_numElements ) ? i : -1;
}

int CPriorityQueue::Parent(int child)
{
	if (child != 0)
    {
        int i = (child - 1) >> 1;
        return i;
    }
    return -1;
}

void CPriorityQueue::HeapifyUp(int index)
{
	while ((index > 0) && (Parent(index) >= 0) && 
		(m_pElements[Parent(index)].value > m_pElements[index].value) )
    {
        SWAP(Parent(index), index);
		index = Parent(index);
    }
}

void CPriorityQueue::HeapifyDown(int index)
{
	int child = Left(index);
    if ( (child > 0) && (Right(index) > 0) &&
		(m_pElements[child].value > m_pElements[Right(index)].value ) )
    {
        child = Right(index);
    }
	if ( child > 0 && m_pElements[index].value > m_pElements[child].value)
    {
        SWAP(index, child);
        HeapifyDown(child);
    }
}

inline void CPriorityQueue::SWAP(int index1, int index2)
{
	ELEMENT tmp = m_pElements[index1];
	m_pElements[index1] = m_pElements[index2];
	m_pElements[index2] = tmp;   
}

void CPriorityQueue::IncreaseHeapSize()
{
	int oldSize = m_Size;
	ELEMENT* pOldElements = m_pElements;
	m_Size *= 2;
	m_pElements = new ELEMENT[m_Size];
	memcpy(m_pElements, pOldElements, oldSize * sizeof(ELEMENT));
	delete [] pOldElements;
}