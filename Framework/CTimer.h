#ifndef _C_TIMER_H_
#define _C_TIMER_H_

#include <iostream>
#include <time.h>
#include <string>

class CTimer
{
public:

	enum TimerType { CPU, OGL};

	CTimer(TimerType type) : m_Time(0.) { m_Type = type; };
	~CTimer() { }

	void Start();
	void Stop();
	void Stop(std::string s);
	
	double GetTime();		// returns the time in milliseconds.
		
private:
	TimerType m_Type;
	
	double m_Time;
	
	clock_t m_ClockStartCPU;
	clock_t m_ClockEndCPU;
};

#endif