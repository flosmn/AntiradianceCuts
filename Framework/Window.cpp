#ifndef _DEBUG
#define _DEBUG
#endif

#pragma comment(linker, "/subsystem:windows")

#include <glm/glm.hpp>

#include <windows.h>

#include <GL/glew.h>
#include <GL/gl.h>
#include <GL/glu.h>

#include <iostream>

#include "CGUI.h"
#include "CConfigManager.h"

#include "Render.h"
#include "Camera.h"
#include "OGLResources\COGLContext.h"
#include "CTimer.h"

#include "Utils\Rand.h"

#include "CTestFramework.h"

#include "guicon.h"

#include <time.h>

typedef unsigned int uint;

HDC g_HDC;
int window_width = 640;
int window_height = 480;
bool fullScreen = false;

Camera* g_pCamera;
Renderer* g_pRenderer;
CConfigManager* g_pConfigManager;
CGUI* g_pGUI;
COGLContext* g_pOGLContext;

uint g_MousePosX = 0;
uint g_MousePosY = 0;

clock_t g_Clock;

bool HandleMouseMoveEvent(WPARAM wParam, LPARAM lParam);
bool HandleKeyEvent(WPARAM wParam);

float CalcFPS();

void Render()
{
	//g_pRenderer->Render();
	g_pRenderer->ClusteringTestRender();

	g_pGUI->Render(CalcFPS());

	SwapBuffers(g_HDC);
}

bool Init()
{
	int t = (int)(time(NULL));
	RandInit(t);
	srand(0);

	GLenum err = glewInit();
	if (GLEW_OK != err)
	{
		std::cout << "Error initializing glew: " << glewGetErrorString(err) << std::endl;
		return false;
	}
	std::cout << "GLEW version: " << glewGetString(GLEW_VERSION) << std::endl;

	std::cout << "OpenGL version: " << glGetString(GL_VERSION) << std::endl;
	std::cout << "GLSL version: " << glGetString(GL_SHADING_LANGUAGE_VERSION) << std::endl;
	
	g_pCamera = new Camera(window_width, window_height, 0.1f, 5000.0f);
	g_pRenderer = new Renderer(g_pCamera);
	g_pConfigManager = new CConfigManager(g_pRenderer);
	g_pGUI = new CGUI(g_pConfigManager);

	g_pRenderer->SetConfigManager(g_pConfigManager);
	g_pRenderer->SetOGLContext(g_pOGLContext);
	g_pConfigManager->Update();

	if(!g_pRenderer->Init())
	{
		std::cout << "Renderer initialization failed." << std::endl;
	}
		
	g_pGUI->Init(window_width, window_height);
	
	return true;
}

void Release()
{
	g_pRenderer->Release();
	g_pGUI->Release();

	delete g_pConfigManager;
	delete g_pRenderer;
	delete g_pCamera;
	delete g_pGUI;
}

void SetupPixelFormat(HDC hDC)
{
	int nPixelFormat;

	static PIXELFORMATDESCRIPTOR pfd = {
		sizeof(PIXELFORMATDESCRIPTOR),          //size of structure
		1,                                      //default version
		PFD_DRAW_TO_WINDOW |                    //window drawing support
		PFD_SUPPORT_OPENGL |                    //opengl support
		PFD_DOUBLEBUFFER,                       //double buffering support
		PFD_TYPE_RGBA,                          //RGBA color mode
		32,                                     //32 bit color mode
		0, 0, 0, 0, 0, 0,                       //ignore color bits
		0,                                      //no alpha buffer
		0,                                      //ignore shift bit
		0,                                      //no accumulation buffer
		0, 0, 0, 0,                             //ignore accumulation bits
		32,                                     //16 bit z-buffer size
		0,                                      //no stencil buffer
		0,                                      //no aux buffer
		PFD_MAIN_PLANE,                         //main drawing plane
		0,                                      //reserved
		0, 0, 0 };                              //layer masks ignored

		nPixelFormat = ChoosePixelFormat(hDC, &pfd);

		/* Set the pixel format to the device context */
		SetPixelFormat(hDC, nPixelFormat, &pfd);
}

/*      Windows Event Procedure Handler
*/
LRESULT CALLBACK WndProc(HWND hwnd, UINT message, WPARAM wParam, LPARAM lParam)
{
	if(g_pGUI != 0)
	{
		if(g_pGUI->HandleEvent(hwnd, message, wParam, lParam))
		{
			g_pConfigManager->Update();
			return 0;
		}
	}
	
	static HDC hDC;

	int width, height;

	switch(message)
	{
	case WM_CREATE: //window being created

		hDC = GetDC(hwnd);  //get current windows device context
		g_HDC = hDC;
		SetupPixelFormat(hDC); //call our pixel format setup function

		/*      Create rendering context and make it current
		*/
		g_pOGLContext = new COGLContext(wglCreateContext(hDC));
		wglMakeCurrent(hDC, g_pOGLContext->GetGLContext());

		return 0;
		break;

	case WM_CLOSE:  //window is closing

		Release();

		/* Deselect rendering context and delete it*/
		wglMakeCurrent(hDC, NULL);
		wglDeleteContext(g_pOGLContext->GetGLContext());

		/* Send quit message to queue*/
		PostQuitMessage(0);

		return 0;
		break;

	case WM_SIZE:

		/* Retrieve width and height*/
		height = HIWORD(lParam);
		width = LOWORD(lParam);

		/* Don't want a divide by 0*/
		if (height == 0)
		{
			height = 1;
		}

		/* Reset the viewport to new dimensions*/
		glViewport(0, 0, width, height);

		return 0;
		break;

	case WM_KEYDOWN:
	
		if(HandleKeyEvent(wParam))
		{
			g_pConfigManager->Update();
			return 0;
		}

		break;
				
	case WM_MOUSEMOVE:

		if(HandleMouseMoveEvent(wParam, lParam))
		{
			g_pConfigManager->Update();
			return 0;
		}

		break;

	default: break;
		
	}

	return (DefWindowProc(hwnd, message, wParam, lParam));
}

int APIENTRY WinMain(HINSTANCE hInstance,
	HINSTANCE hPrevInstance,
	LPSTR     lpCmdLine,
	int       nCmdShow)
{
	#ifdef _DEBUG
	RedirectIOToConsole();
	#endif
	
	WNDCLASSEX windowClass;		//window class
	HWND    hwnd;               //window handle
	MSG     msg;                //message
	bool    done;               //flag for completion of app
	DWORD   dwExStyle;          //window extended style
	DWORD   dwStyle;            //window style
	RECT    windowRect;

	/* Screen/display attributes */
	int bits = 32;

	windowRect.left		= (long)0;              //set left value to 0
	windowRect.right	= (long)window_width;	//set right value to requested width
	windowRect.top		= (long)0;              //set top value to 0
	windowRect.bottom	= (long)window_height;	//set bottom value to requested height

	/*      Fill out the window class structure*/
	windowClass.cbSize			= sizeof(WNDCLASSEX);
	windowClass.style			= CS_HREDRAW | CS_VREDRAW;
	windowClass.lpfnWndProc		= WndProc;
	windowClass.cbClsExtra		= 0;
	windowClass.cbWndExtra		= 0;
	windowClass.hInstance		= hInstance;
	windowClass.hIcon			= LoadIcon(NULL, IDI_APPLICATION);
	windowClass.hCursor			= LoadCursor(NULL, IDC_ARROW);
	windowClass.hbrBackground	= NULL;
	windowClass.lpszMenuName	= NULL;
	windowClass.lpszClassName	= "AntiradianceCuts";
	windowClass.hIconSm			= LoadIcon(NULL, IDI_WINLOGO);

	/* Register window class*/
	if (!RegisterClassEx(&windowClass))
	{
		return 0;
	}

	/* Check if fullscreen is on*/
	if (fullScreen)
	{
		DEVMODE dmScreenSettings;
		memset(&dmScreenSettings, 0, sizeof(dmScreenSettings));
		dmScreenSettings.dmSize = sizeof(dmScreenSettings);
		dmScreenSettings.dmPelsWidth = window_width;
		dmScreenSettings.dmPelsHeight = window_height;
		dmScreenSettings.dmBitsPerPel = bits;
		dmScreenSettings.dmFields = DM_BITSPERPEL | DM_PELSWIDTH | DM_PELSHEIGHT;

		if (ChangeDisplaySettings(&dmScreenSettings, CDS_FULLSCREEN !=
			DISP_CHANGE_SUCCESSFUL))
		{
			/* Setting display mode failed, switch to windowed*/
			MessageBox(NULL, "Display mode failed", NULL, MB_OK);
			fullScreen = false;
		}
	}

	/* Check if fullscreen is still on*/
	if (fullScreen)
	{
		dwExStyle = WS_EX_APPWINDOW; //window extended style
		dwStyle = WS_POPUP;          //windows style
		ShowCursor(FALSE);           //hide mouse pointer
	}

	else
	{
		dwExStyle = WS_EX_APPWINDOW | WS_EX_WINDOWEDGE;
		dwStyle = WS_OVERLAPPEDWINDOW;
	}

	AdjustWindowRectEx(&windowRect, dwStyle, FALSE, dwExStyle);

	/* Class registerd, so now create our window*/
	hwnd = CreateWindowEx(NULL, "AntiradianceCuts",  //class name
		"AntiradianceCuts",
		dwStyle |
		WS_CLIPCHILDREN |
		WS_CLIPSIBLINGS,
		0, 0, //x and y coords
		windowRect.right - windowRect.left,
		windowRect.bottom - windowRect.top,
		NULL,										// handle to parent
		NULL,										// handle to menu
		hInstance,									// application instance
		NULL);										// no xtra params

	/* Check if window creation failed (hwnd = null ?)*/
	if (!hwnd)
	{
		return 0;
	}
	
	if (!Init())
	{
		return 0;
	}
	
	ShowWindow(hwnd, SW_SHOW);	//display window
	UpdateWindow(hwnd);			//update window
	
	g_Clock = clock();

	done = false;   //initialize loop condition variable
	
	// run tests before rendering
	CTestFramework testFramework;
	//testFramework.RunTests();
	
	/* Main message loop*/
	while (!done)
	{
		PeekMessage(&msg, hwnd, NULL, NULL, PM_REMOVE);

		if (msg.message == WM_QUIT)
		{
			done = true;
		}

		else
		{
			Render();
			TranslateMessage(&msg);
			DispatchMessage(&msg);
		}
	}

	if (fullScreen)
	{
		ChangeDisplaySettings(NULL, 0);
		ShowCursor(TRUE);
	}

	return (int)msg.wParam;
}

bool HandleMouseMoveEvent(WPARAM wParam, LPARAM lParam)
{
	int x=(short)LOWORD(lParam);
	int y=(short)HIWORD(lParam);

	int deltaX = x - g_MousePosX;
	int deltaY = y - g_MousePosY;

	g_MousePosX = x;
	g_MousePosY = y;

	uint leftButtonDown = wParam & MK_LBUTTON;
	uint rightButtonDown = wParam & MK_RBUTTON;
		
	if(leftButtonDown) {
		g_pCamera->RotLeft((float)deltaX/window_width);
		g_pCamera->RotUp((float)deltaY/window_height);
		g_pRenderer->ClearAccumulationBuffer();
		return true;
	}
	else if(rightButtonDown) {
		g_pCamera->ZoomIn((float)deltaY/window_height);
		g_pRenderer->ClearAccumulationBuffer();
		return true;
	}

	return false;
}

bool HandleKeyEvent(WPARAM wParam)
{
	switch(wParam)
    {
		case 'E':

			g_pRenderer->Export();
			return true; break;

		case 'T':

			g_pConfigManager->GetConfVarsGUI()->DrawDebugTextures = !g_pConfigManager->GetConfVars()->DrawDebugTextures;
			return true; break;

		case 'X':

			g_pConfigManager->GetConfVarsGUI()->UseToneMapping = !g_pConfigManager->GetConfVars()->UseToneMapping;
			return true; break;

		case 'C':

			g_pRenderer->ClearLighting();
			return true; break;

		case 'V':

			g_pRenderer->NewDebugLights();
			return true; break;

		case 'M':

			g_pCamera->ZoomIn(0.5f);
			g_pRenderer->ClearLighting();
			return true; break;

		case 'N':

			g_pCamera->ZoomOut(0.5f);
			g_pRenderer->ClearLighting();
			return true; break;
			
		default: break;
    }
	
	return false;
}

float CalcFPS()
{
	clock_t delta;
	do {
		delta = clock() - g_Clock;
	} while (delta == 0);

	g_Clock = clock();

	return (float)((double)CLOCKS_PER_SEC / (double)delta);
}