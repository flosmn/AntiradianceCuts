#ifndef _C_GUI_H_
#define _C_GUI_H_

#include "AntTweakBar.h"

class CConfigManager;

typedef unsigned int uint;

class CGUI
{
public:
	CGUI(CConfigManager* pConfigManager);
	~CGUI();

	bool Init(uint window_width, uint window_height);
	void Release();

	bool HandleEvent(void* wnd, uint msg, uint wParam, uint lParam);
	void Render(float fps);

private:
	TwBar* m_pTwBar;

	float m_fps;

	CConfigManager* m_pConfigManager;
};

#endif // _C_GUI_H_