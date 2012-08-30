#include "CExperimentData.h"

#include <iostream>
#include <fstream>
#include <sstream>

CExperimentData::CExperimentData()
{
	m_MaxPaths = 0;
	m_MaxTime = 0.f;
	m_Written = false;
}

CExperimentData::~CExperimentData()
{
	m_Data.clear();
}

void CExperimentData::Init(std::string name, std::string filename)
{
	m_Name = name;
	m_FileName = filename;
}

void CExperimentData::ClearData()
{
	m_Data.clear();
}

void CExperimentData::AddData(int nPaths, float time, float error)
{
	STAMP s;
	s.nPaths = nPaths;
	s.time = time;
	s.error = error;
	m_Data.push_back(s);

	if(m_MaxPaths > 0 && nPaths >= m_MaxPaths && !m_Written)
		WriteToFile();
	else if(m_MaxTime > 0.f && time >= m_MaxTime && !m_Written)
		WriteToFile();
}

void CExperimentData::WriteToFile()
{
	std::stringstream path;
	path << "Experiments/" << m_FileName;

	std::stringstream content;
	content << "nPaths time " << m_Name << "\n";
	for(int i = 0; i < m_Data.size(); ++i)
	{
		STAMP s = m_Data[i];
		content << s.nPaths << " " << s.time << " " << s.error << "\n";
	}

	std::ofstream dataFile;
	dataFile.open (path.str());
	
	dataFile << content.str();
	dataFile.close();

	m_Written = true;
}