#ifndef _C_EXPERIMENT_DATA_H_
#define _C_EXPERIMENT_DATA_H_

#include <vector>
#include <string>

struct STAMP
{
	int nPaths;
	float time; // in sec.
	float error;
};

class CExperimentData 
{
public:
	CExperimentData();
	~CExperimentData();

	void Init(std::string name, std::string filename);

	void ClearData();

	void AddData(int nPaths, float time, float error);

	void MaxTime(float t) { m_MaxTime = t; }
	void MaxPaths(int n) { m_MaxPaths = n; }
	
private:
	void WriteToFile();
	
	std::string m_Name;
	std::string m_FileName;

	float m_MaxTime;
	int m_MaxPaths;
	bool m_Written;

	std::vector<STAMP> m_Data;
};

#endif // _C_EXPERIMENT_DATA_H_