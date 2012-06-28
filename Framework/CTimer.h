#ifndef _CTIMER_H_
#define _CTIMER_H_

#include <gl\glew.h>

#include <string>
#include <iostream>
#include <time.h>

class CTimer
{
public:
	enum TIMERTYPE { GPU, CPU };

	CTimer(TIMERTYPE type);
	~CTimer();

	void Start();
	void Stop();
	
	double GetTime();		// returns the time in milliseconds.
		
private:
	void WaitTillAvailable(GLuint query);

	TIMERTYPE m_Type;
	double m_Time;

	GLuint m_QueryStartGPU;
	GLuint m_QueryEndGPU;
	
	GLuint m_TimeStartGPU;
	GLuint m_TimeEndGPU;

	clock_t m_ClockStartCPU;
	clock_t m_ClockEndCPU;
};

#endif