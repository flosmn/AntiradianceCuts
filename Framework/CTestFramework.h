#ifndef _C_TEST_FRAMEWORK_H_
#define _C_TEST_FRAMEWORK_H_

class CTestFramework
{
public:
	CTestFramework();
	~CTestFramework();

	void RunTests();

private:
	void TriangleIntersectionTest();
	void KdTreeBuildTest();
};

#endif _C_TEST_FRAMEWORK_H_