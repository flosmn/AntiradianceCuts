#include "CTimer.h"

#include "gl/glew.h"

void CTimer::Start()
{
	if(m_Type == OGL)
	{
		glFinish();
	}
	
	m_ClockStartCPU = clock();
}

void CTimer::Stop()
{
	if(m_Type == OGL)
	{
		glFinish();
	}
	

	m_ClockEndCPU = clock();
	clock_t diff = m_ClockEndCPU - m_ClockStartCPU;
	
	m_Time = 1000.0 * double(diff) / (double(CLOCKS_PER_SEC));	
}

void CTimer::Stop(std::string s)
{
	Stop();

	std::cout << s << " took " << m_Time << "ms." << std::endl;
}

double CTimer::GetTime()
{
	Stop(); 

	return m_Time;
}