#include "CTimer.h"
#include <time.h>

CTimer::CTimer(GLuint maxEvents)
{
	m_MaxEvents = maxEvents + 1;
	m_Index = 0;
}

void CTimer::Init()
{
	m_Queries = new GLuint[2 * m_MaxEvents];
	m_TimesStartGPU = new GLuint[m_MaxEvents];
	m_TimesEndGPU = new GLuint[m_MaxEvents];
	m_TimesStartCPU = new time_t[m_MaxEvents];
	m_TimesEndCPU = new time_t[m_MaxEvents];
	m_NumberOfStartStops = new GLuint[m_MaxEvents];
	m_TotalTime = new unsigned long[m_MaxEvents];

	Reset();

	m_Index = 1;

	glGenQueries(m_MaxEvents, m_Queries);
}

CTimer::~CTimer()
{
	glDeleteQueries(m_MaxEvents, m_Queries);

	delete [] m_Queries;
	delete [] m_TimesStartGPU;
	delete [] m_TimesEndGPU;
	delete [] m_TimesStartCPU;
	delete [] m_TimesEndCPU;
	delete [] m_NumberOfStartStops;
	delete [] m_TotalTime;
}

void CTimer::RegisterEvent(std::string eventName, TIMERTYPE type)
{
	if (m_mapEventNameToIndex[eventName] != NULL) {
		std::cout << "Event " << eventName << " already registered." << std::endl;
	}
	else if (m_Index >= m_MaxEvents) {
		std::cout << "Maximum number of registerable events reached."  << std::endl;
	}
	else {
		m_mapEventNameToIndex[eventName] = m_Index;
		m_mapEventIndexToName[m_Index] = eventName;
		m_mapEventNameToType[eventName] = type;
		m_Index++;
	}
}

void CTimer::StartEvent(std::string eventName)
{
	GLuint index = m_mapEventNameToIndex[eventName];
	if (index == NULL) {
		std::cout << "Event " << eventName << " not registered." << std::endl;
	}
	else {
		if(m_mapEventNameToType[eventName] == GPU)
		{
			glQueryCounter(m_Queries[2 * index + 0], GL_TIMESTAMP);
			m_NumberOfStartStops[index]++;
		}
		else
		{
			m_TimesStartCPU[index] = time(NULL);
			m_NumberOfStartStops[index]++;
		}
	}	
}

void CTimer::StopEvent(std::string eventName)
{
	GLuint index = m_mapEventNameToIndex[eventName];
	if (index == NULL) {
		std::cout << "Event " << eventName << " not registered." << std::endl;
	}
	else {
		if (m_mapEventNameToType[eventName] == GPU)
		{
			glQueryCounter(m_Queries[2 * index + 1], GL_TIMESTAMP);
		
			WaitTillAvailable(m_Queries[2 * index + 0]);
			WaitTillAvailable(m_Queries[2 * index + 1]);
		
			glGetQueryObjectuiv(m_Queries[2 * index + 0], GL_QUERY_RESULT, &m_TimesStartGPU[index]);
			glGetQueryObjectuiv(m_Queries[2 * index + 1], GL_QUERY_RESULT, &m_TimesEndGPU[index]);
		
			GLuint time = (m_TimesEndGPU[index] - m_TimesStartGPU[index])/1000;
			if (time < 0) {
				std::cout << "Measured time <0 !" << std::endl;
			}
			unsigned long temp = m_TotalTime[index];
			m_TotalTime[index] += time;
			if(m_TotalTime[index] < temp){
				std::cout << "total time overflow!" << std::endl;
			}
		}
		else
		{
			m_TimesEndCPU[index] = time(NULL);
			double diff = difftime(m_TimesEndCPU[index], m_TimesStartCPU[index]);
			GLuint time = GLuint(diff * 1000 * 1000);
			if (time < 0) {
				std::cout << "Measured time <0 !" << std::endl;
			}
			unsigned long temp = m_TotalTime[index];
			m_TotalTime[index] += time;
			if(m_TotalTime[index] < temp){
				std::cout << "total time overflow!" << std::endl;
			}
		}
	}	
}

float CTimer::GetTime(std::string eventName)
{
	float time = 0;
	GLuint index = m_mapEventNameToIndex[eventName];
	if (index == NULL) {
		std::cout << "Event " << eventName << " not registered." << std::endl;
	}
	else {
		// convert to ms
		time = ((float)m_TotalTime[index] / (1000*(float)m_NumberOfStartStops[index])); 
	}
	
	return time;
}

void CTimer::PrintStats()
{
	std::cout << "Print timer stats:" << std::endl;
	for(int i = 1; i < m_Index; ++i)
	{
		// convert to ms
		std::cout << "\t event: " << m_mapEventIndexToName[i]; // << std::endl;
		//std::cout << "\t \t total time: " << m_TotalTime[i] << std::endl;
		//std::cout << "\t \t # of measures: " << m_NumberOfStartStops[i] << std::endl;
		float time = ((float)m_TotalTime[i] / (1000*(float)m_NumberOfStartStops[i])); 
		std::cout << "\t \t time: " << time << " ms." << std::endl;
	}
}

void CTimer::Reset()
{
	for (int i = 0; i < m_MaxEvents; ++i) {
		m_NumberOfStartStops[i] = 0;
		m_TotalTime[i] = 0;
	}
}

void CTimer::WaitTillAvailable(GLuint query)
{
	GLuint available = 0;
    while (!available) {
		glGetQueryObjectuiv(query, GL_QUERY_RESULT_AVAILABLE, &available);
    }
}