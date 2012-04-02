#ifndef _CTIMER_H_
#define _CTIMER_H_

#include <gl\glew.h>

#include <map>
#include <string>
#include <iostream>

/*
	this class implement simple OpenGL timing queries. Every StartEvent pushes a new event onto
	an event stack and with EndEvent the topmost event is poped from the event stack and the
	time from start to end is stored and can be queried with the function GetTime.
	To print the time of all events onto console call PrintStats.
*/
class CTimer
{
public:
	enum TIMERTYPE { GPU, CPU };

	CTimer(GLuint maxEvents);
	~CTimer();

	void Init();
	void RegisterEvent(std::string eventName, TIMERTYPE type);
	void StartEvent(std::string eventName);		// start the timer for an event. multiple start-stop sequences are averaged
	void StopEvent(std::string eventName);		// stops the timer for an event. can be started again
	void Reset();								// resets all times mesuared by the timer. registerd events are not lost

	float GetTime(std::string eventName);		// returns the time in milliseconds.
	void PrintStats();
private:
	void WaitTillAvailable(GLuint query);

	GLuint* m_Queries;
	
	GLuint* m_TimesStartGPU;
	GLuint* m_TimesEndGPU;

	time_t* m_TimesStartCPU;
	time_t* m_TimesEndCPU;

	GLuint* m_NumberOfStartStops;		// tracks the number of times a start-stop sequence has been called for an event
	unsigned long* m_TotalTime;			// tracks the total time for a registered event
	
	int m_MaxEvents;
	int m_Index;

	std::map<std::string, GLuint> m_mapEventNameToIndex;
	std::map<GLuint, std::string> m_mapEventIndexToName;
	std::map<std::string, TIMERTYPE> m_mapEventNameToType;
};

#endif