#include "CTimer.h"
#include <time.h>

CTimer::CTimer(TIMERTYPE type)
{	
	m_Type = type;
	m_Time = 0;
	glGenQueries(1, &m_QueryStartGPU);
	glGenQueries(1, &m_QueryEndGPU);
}

CTimer::~CTimer()
{
	glDeleteQueries(1, &m_QueryStartGPU);
	glDeleteQueries(1, &m_QueryEndGPU);
}

void CTimer::Start()
{
	if(m_Type == GPU)
	{
		glQueryCounter(m_QueryStartGPU, GL_TIMESTAMP);
	}
	else
	{
		m_ClockStartCPU = clock();
	}	
}

void CTimer::Stop()
{
	if (m_Type == GPU)
	{
		glQueryCounter(m_QueryEndGPU, GL_TIMESTAMP);
		
		WaitTillAvailable(m_QueryStartGPU);
		WaitTillAvailable(m_QueryEndGPU);
		
		glGetQueryObjectuiv(m_QueryStartGPU, GL_QUERY_RESULT, &m_TimeStartGPU);
		glGetQueryObjectuiv(m_QueryEndGPU, GL_QUERY_RESULT, &m_TimeEndGPU);
		
		GLuint time = (m_TimeEndGPU - m_TimeStartGPU)/1000;
	}
	else
	{
		m_ClockEndCPU = clock();
		clock_t diff = m_ClockEndCPU - m_ClockStartCPU;
		if (diff < 0) {
			std::cout << "Measured #clocks <0 !" << std::endl;
		}
		else
		{
			m_Time = 1000.0 * double(diff) / (double(CLOCKS_PER_SEC));
		}
	}	
}

double CTimer::GetTime()
{
	return m_Time;
}

void CTimer::WaitTillAvailable(GLuint query)
{
	GLuint available = 0;
    while (!available) {
		glGetQueryObjectuiv(query, GL_QUERY_RESULT_AVAILABLE, &available);
    }
}