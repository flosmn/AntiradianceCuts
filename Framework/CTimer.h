#ifndef _C_TIMER_H_
#define _C_TIMER_H_

#include <iostream>
#include <time.h>
#include <string>

class COCLContext;

class CTimer
{
public:

	enum TimerType { CPU, OGL, OCL};

	CTimer(TimerType type, COCLContext* pContext = 0) : m_Time(0.), m_pContext(pContext) { m_Type = type; };
	~CTimer() { }

	void Start();
	void Stop();
	void Stop(std::string s);
	
	double GetTime();		// returns the time in milliseconds.
		
private:
	TimerType m_Type;

	
	double m_Time;
	
	COCLContext* m_pContext;

	clock_t m_ClockStartCPU;
	clock_t m_ClockEndCPU;
};

#endif