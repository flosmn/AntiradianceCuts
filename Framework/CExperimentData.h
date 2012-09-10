#ifndef _C_EXPERIMENT_DATA_H_
#define _C_EXPERIMENT_DATA_H_

#include <vector>
#include <string>

struct STAMP
{
	int nAVPLs;
	float time; // in sec.
	float error;
	float AVPLsPerSecond;
};

class CExperimentData 
{
public:
	CExperimentData();
	~CExperimentData();

	void Init(std::string name, std::string filename);

	void ClearData();

	void AddData(int nAVPLs, float time, float error, float AVPLsPerSecond);

	void MaxTime(float t) { m_MaxTime = t; }
	void MaxAVPLs(int n) { m_MaxAVPLs = n; }
	
	void WriteToFile();

private:	
	std::string m_Name;
	std::string m_FileName;

	float m_MaxTime;
	int m_MaxAVPLs;
	bool m_Written;

	std::vector<STAMP> m_Data;
};

#endif // _C_EXPERIMENT_DATA_H_