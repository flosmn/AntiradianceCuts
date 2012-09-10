#include "CExperimentData.h"

#include <iostream>
#include <fstream>
#include <sstream>

CExperimentData::CExperimentData()
{
	m_MaxAVPLs = 0;
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

void CExperimentData::AddData(int nAVPLs, float time, float error, float AVPLsPerSecond)
{
	STAMP s;
	s.nAVPLs = nAVPLs;
	s.time = time;
	s.error = error;
	s.AVPLsPerSecond = AVPLsPerSecond;
	m_Data.push_back(s);

	if(m_MaxAVPLs > 0 && nAVPLs >= m_MaxAVPLs && !m_Written)
		WriteToFile();
	else if(m_MaxTime > 0.f && time >= m_MaxTime && !m_Written)
		WriteToFile();
}

void CExperimentData::WriteToFile()
{
	std::stringstream path;
	path << "Experiments/" << m_FileName;

	std::stringstream content;
	content << "AVPLs time error avpls_per_second\n";
	for(int i = 0; i < m_Data.size(); ++i)
	{
		STAMP s = m_Data[i];
		content << s.nAVPLs << " " << s.time << " " << s.error << " " << s.AVPLsPerSecond << "\n";
	}

	std::ofstream dataFile;
	dataFile.open (path.str());
	
	dataFile << content.str();
	dataFile.close();

	m_Written = true;

	std::cout << "Data exported" << std::endl;
}