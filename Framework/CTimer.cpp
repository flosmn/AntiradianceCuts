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
	m_ClocksStartCPU = new clock_t[m_MaxEvents];
	m_ClocksEndCPU = new clock_t[m_MaxEvents];
	m_NumberOfStartStops = new GLuint[m_MaxEvents];
	m_TotalClocks = new clock_t[m_MaxEvents];

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
	delete [] m_ClocksStartCPU;
	delete [] m_ClocksEndCPU;
	delete [] m_NumberOfStartStops;
	delete [] m_TotalClocks;
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
			m_ClocksStartCPU[index] = clock();
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
			/*
			// does not work
			double temp = m_TotalTime[index];
			m_TotalTime[index] += time;
			if(m_TotalTime[index] < temp){
				std::cout << "total time overflow!" << std::endl;
			}
			*/
		}
		else
		{
			m_ClocksEndCPU[index] = clock();
			clock_t diff = m_ClocksEndCPU[index] - m_ClocksStartCPU[index];
			if (diff < 0) {
				std::cout << "Measured #clocks <0 !" << std::endl;
			}
			clock_t temp = m_TotalClocks[index];
			m_TotalClocks[index] += diff;
			if(m_TotalClocks[index] < temp) {
				std::cout << "total #clocks overflow!" << std::endl;
			}
		}
	}	
}

double CTimer::GetTime(std::string eventName)
{
	float time = 0;
	GLuint index = m_mapEventNameToIndex[eventName];
	if (index == NULL) {
		std::cout << "Event " << eventName << " not registered." << std::endl;
	}
	return GetTime(index);
}

double CTimer::GetTime(int eventIndex)
{
	clock_t numClocks = m_TotalClocks[eventIndex];
	double time_total = double(numClocks) / double(CLOCKS_PER_SEC);
	double time_avg = time_total / m_NumberOfStartStops[eventIndex];

	// convert to ms
	return time_avg * 1000; 
}

void CTimer::PrintStats()
{
	std::cout << "Print timer stats:" << std::endl;
	for(int i = 1; i < m_Index; ++i)
	{
		std::cout << "\t event: " << m_mapEventIndexToName[i] << std::endl;
		std::cout << "\t \t total time: " << double(m_TotalClocks[i]) / double(CLOCKS_PER_SEC) << std::endl;
		std::cout << "\t \t # of measures: " << m_NumberOfStartStops[i] << std::endl;
		std::cout << "\t \t time: " << GetTime(i) << " ms." << std::endl;
	}
}

void CTimer::Reset()
{
	for (int i = 0; i < m_MaxEvents; ++i) {
		m_NumberOfStartStops[i] = 0;
		m_TotalClocks[i] = 0;
	}
}

void CTimer::WaitTillAvailable(GLuint query)
{
	GLuint available = 0;
    while (!available) {
		glGetQueryObjectuiv(query, GL_QUERY_RESULT_AVAILABLE, &available);
    }
}