#include "Intersection.h"

#include "CTriangle.h"

glm::vec3 Intersection::GetNormal() const {
	return m_pPrimitive->GetNormal(); 
}

uint Intersection::GetMaterialIndex() const { 
	return m_pPrimitive->GetMaterialIndex(); 
}