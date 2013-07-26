#ifndef TIMER_H_
#define TIMER_H_

#include <chrono>

class Timer
{
	public:
		Timer() {};
		~Timer() {};

		void start()
		{
			m_start = std::chrono::high_resolution_clock::now();
		}
		void stop()
		{
			m_stop = std::chrono::high_resolution_clock::now();
		}

		int getElapsedTimeInSeconds()
		{
			return std::chrono::duration_cast<std::chrono::seconds>
				(std::chrono::high_resolution_clock::now() - m_start).count();
		}

		int getElapsedTimeInMilliseconds()
		{
			return std::chrono::duration_cast<std::chrono::milliseconds>
				(std::chrono::high_resolution_clock::now() - m_start).count();
		}

		int getTimeInSeconds()
		{
			return std::chrono::duration_cast<std::chrono::seconds>
				(m_stop - m_start).count();
		}

		int getTimeInMilliseconds()
		{
			return std::chrono::duration_cast<std::chrono::milliseconds>
				(m_stop - m_start).count();
		}

	private:
		std::chrono::time_point<std::chrono::high_resolution_clock> m_start;
		std::chrono::time_point<std::chrono::high_resolution_clock> m_stop;
};

#endif
