/*
#define GLFW_DLL
#include <GL/glew.h>
#include <GL/glfw.h>

#include "AntTweakBar.h"

#include "Defines.h"
#include "Macros.h"

#include "Utils\GLErrorUtil.h"
#include "Utils\Rand.h"

#include "Render.h"
#include "Camera.h"

#include "CConfigManager.h"

#include "OGLResources\COGLResource.h"
#include "OGLResources\COGLBindLock.h"
#include "OGLResources\COGLTexture2D.h"

#include "MeshResources\CObjFileLoader.h"
#include "MeshResources\CMeshGeometry.h"
#include "MeshResources\CMeshMaterial.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <memory>
#include <time.h>

using std::cout;
using std::cin;
using std::endl;

int windowWidth = 1280;
int	windowHeight = 720;

Renderer *renderer = 0;
Camera* camera = 0;
CConfigManager* configManager = 0;

bool updateConfig = false;
void CleanUp();
void Quit(int status);
void Init();
//void Release();
void GLFWCALL WindowResize(int width, int height);
void GLFWCALL KeyCallback(int key, int action);
void GLFWCALL MouseButtonCallback(int key, int action);
void GLFWCALL MousePositionCallback(int x, int y);
void GLFWCALL MouseWheelCallback(int pos);
int GLFWCALL WindowClose(void);

glm::vec2 mousePosition;
bool leftMouseButton = false;
bool rightMouseButton = false;

double fps;

int abs()
{
	int t = (int)(time(NULL));
	RandInit(t);
	
	GLFWvidmode mode;
	
	int running = GL_TRUE;

	if(glfwInit() == GL_FALSE) {
		cout << "glfwInit failed." << endl;
		Quit(EXIT_FAILURE);
	}
		
	glfwOpenWindowHint(GLFW_OPENGL_VERSION_MAJOR, 4);
	glfwOpenWindowHint(GLFW_OPENGL_VERSION_MINOR, 2);	
	glfwOpenWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_COMPAT_PROFILE);

	glfwGetDesktopMode(&mode);
	if(glfwOpenWindow(windowWidth, windowHeight, mode.RedBits, mode.GreenBits, mode.BlueBits, 0, 32, 32, GLFW_WINDOW) == GL_FALSE) {
		cout << "glfwOpenWindow failed." << endl;
		Quit(EXIT_FAILURE);
	}

	Init();

	cout << "Version: " << glGetString(GL_VERSION) << endl;
	cout << "Shader Version: " << glGetString(GL_SHADING_LANGUAGE_VERSION) << endl;
	
	GLenum err = glewInit();
	if (GLEW_OK != err)
	{
		fprintf(stderr, "Error: %s\n", glewGetErrorString(err));
		glfwTerminate();
		exit( EXIT_FAILURE );
	}
	fprintf(stdout, "Status: Using GLEW %s\n", glewGetString(GLEW_VERSION));

	TwInit(TW_OPENGL, NULL);
	TwWindowSize(windowWidth, windowHeight);

	camera = new Camera(windowWidth, windowHeight, 0.1f, 2000.0f); 
	renderer = new Renderer(camera);
	configManager = new CConfigManager(renderer);
	renderer->SetConfigManager(configManager);

	TwBar *myBar;
	myBar = TwNewBar("GUI");
		
	TwAddVarRO(myBar, "fps", TW_TYPE_DOUBLE, &fps, "");
	
	TwAddSeparator(myBar, "", "");

	TwAddVarRW(myBar, "Use Antiradiance", TW_TYPE_INT32, &(configManager->GetConfVarsGUI()->UseAntiradiance), "min=0 max=1 step=1");
	TwAddVarRW(myBar, "Gather With AVPL Atlas", TW_TYPE_INT32, &(configManager->GetConfVarsGUI()->GatherWithAVPLAtlas), "min=0 max=1 step=1");
	TwAddVarRW(myBar, "Filter AVPL Atlas", TW_TYPE_INT32, &(configManager->GetConfVarsGUI()->FilterAvplAtlasLinear), " min=0 max=1 step=1 ");

	TwAddSeparator(myBar, "", "");
	
	TwAddVarRW(myBar, "#Paths", TW_TYPE_INT32, &(configManager->GetConfVarsGUI()->NumPaths), " min=1 max=10000000000 step=1 ");
	TwAddVarRW(myBar, "#Paths per frame", TW_TYPE_INT32, &(configManager->GetConfVarsGUI()->NumPathsPerFrame), " min=1 max=5000 step=1 ");
	TwAddVarRW(myBar, "Cone Factor", TW_TYPE_INT32, &(configManager->GetConfVarsGUI()->ConeFactor), " min=1 max=1000 step=1 ");
	TwAddVarRW(myBar, "#add. AVPL", TW_TYPE_INT32, &(configManager->GetConfVarsGUI()->NumAdditionalAVPLs), " min=0 max=512 step=1 ");
	
	TwAddSeparator(myBar, "", "");
	
	TwAddVarRW(myBar, "Draw AVPL atlas", TW_TYPE_INT32, &(configManager->GetConfVarsGUI()->DrawAVPLAtlas), "min=0 max=1 step=1");
	TwAddVarRW(myBar, "Use Debug Mode", TW_TYPE_INT32, &(configManager->GetConfVarsGUI()->UseDebugMode), "min=0 max=1 step=1");
	TwAddVarRW(myBar, "Draw LOL", TW_TYPE_INT32, &(configManager->GetConfVarsGUI()->DrawLightingOfLight), " min=-1 max=100 step=1 ");
	TwAddVarRW(myBar, "Draw Lights", TW_TYPE_INT32, &(configManager->GetConfVarsGUI()->DrawLights), " min=0 max=1 step=1 ");
	TwAddVarRW(myBar, "RenderBounce", TW_TYPE_INT32, &(configManager->GetConfVarsGUI()->RenderBounce), " min=-1 max=100 step=1 ");
	TwAddVarRW(myBar, "Geo-Term Limit", TW_TYPE_FLOAT, &(configManager->GetConfVarsGUI()->GeoTermLimit), "min=0 max=100000 step=0.00001");
	TwAddVarRW(myBar, "Sqrt # Atlas Samles", TW_TYPE_INT32, &(configManager->GetConfVarsGUI()->NumSqrtAtlasSamples), " min=1 max=100 step=1 ");

	TwAddSeparator(myBar, "", "");

	TwAddVarRW(myBar, "Use Tone Mapping", TW_TYPE_INT32, &(configManager->GetConfVarsGUI()->UseToneMapping), "min=0 max=1 step=1");
	TwAddVarRW(myBar, "Gamma", TW_TYPE_FLOAT, &(configManager->GetConfVarsGUI()->Gamma), " min=0.0 max=10.0 step=0.1 ");
	TwAddVarRW(myBar, "Exposure", TW_TYPE_FLOAT, &(configManager->GetConfVarsGUI()->Exposure), " min=0.0 max=10.0 step=0.1 ");
	
	if(!renderer->Init()) {
		cout << "renderer initialization failed." << endl;
		Quit(EXIT_FAILURE);
	}
		
	configManager->Update();
	
	clock_t c1, c2, delta;
	
	c1 = clock();
	
	while(running) {
		do {
		c2 = clock();
		delta = c2 - c1;
		} while (delta == 0);
				
		fps = (double)CLOCKS_PER_SEC / (double)delta;
		c1 = c2;
		
		if(updateConfig) configManager->Update();
		
		renderer->Render();
		
		TwDraw();

		glfwSwapBuffers();

		running = glfwGetWindowParam(GLFW_OPENED);
	}

	int quit = 0;
	cout << "Quit: " << endl;
	cin >> quit;

	Quit(EXIT_SUCCESS);
}


void Release()
{
	renderer->Release();
}


void Quit(int status)
{
	CleanUp();
	int end;
	std::cin >> end;

	glfwTerminate();
	exit(status);
}

void Init()
{
	glfwSetWindowSizeCallback(WindowResize);
	glfwSetWindowCloseCallback(WindowClose);
	glfwSetKeyCallback(KeyCallback);
	glfwSetMouseButtonCallback(MouseButtonCallback);
	glfwSetMousePosCallback(MousePositionCallback);	
	glfwSetMouseWheelCallback(MouseWheelCallback);	
	glfwSetCharCallback((GLFWcharfun)TwEventCharGLFW);
}

void CleanUp()
{
	SAFE_DELETE(renderer);
}

int GLFWCALL WindowClose(void)
{
	CheckGLError("main", "WindowCLose() before Release()");

	//Release();

	CheckGLError("main", "WindowCLose() after Release()");

	return GL_TRUE;
}

void GLFWCALL WindowResize(int width, int height)
{
	windowWidth = width;
	windowHeight = height;

	if(camera) 
	{
		camera->SetWidth(width);
		camera->SetHeight(height);

		renderer->WindowChanged();
		renderer->ClearLighting();
	}

	glViewport(0, 0, (GLsizei)windowWidth, (GLsizei)windowHeight);
}

void GLFWCALL KeyCallback(int key, int action) 
{	
	updateConfig = true;

	if(TwEventKeyGLFW(key, action)) return;
	
	if(key == GLFW_KEY_ESC && action == GLFW_PRESS){
		Quit(EXIT_SUCCESS);
	}
	if(key == GLFW_KEY_RIGHT && action == GLFW_PRESS){
		camera->RotRight(PI/8.0f);
		renderer->ClearAccumulationBuffer();
	}
	if(key == GLFW_KEY_LEFT && action == GLFW_PRESS){
		camera->RotLeft(PI/8.0f);
		renderer->ClearAccumulationBuffer();
	}
	if(key == GLFW_KEY_UP && action == GLFW_PRESS){
		camera->RotUp(PI/8.0f);
		renderer->ClearAccumulationBuffer();
	}
	if(key == GLFW_KEY_DOWN && action == GLFW_PRESS){
		camera->RotDown(PI/8.0f);
		renderer->ClearAccumulationBuffer();
	}

	if(key == 'E' && action == GLFW_PRESS){
		renderer->Export();
	}

	if(key == 'T' && action == GLFW_PRESS){
		configManager->GetConfVarsGUI()->DrawDebugTextures = !configManager->GetConfVars()->DrawDebugTextures;
	}

	if(key == 'X' && action == GLFW_PRESS){
		configManager->GetConfVarsGUI()->UseToneMapping = !configManager->GetConfVars()->UseToneMapping;
	}
		
	if(key == 'C' && action == GLFW_PRESS){
		renderer->ClearLighting();
	}

	if(key == 'V' && action == GLFW_PRESS){
		renderer->NewDebugLights();
	}

	if(key == 'S' && action == GLFW_PRESS){
		renderer->Stats();
	}

	if(key == 'A' && action == GLFW_PRESS){
		configManager->GetConfVarsGUI()->UseAntiradiance = !configManager->GetConfVars()->UseAntiradiance;
	}

	if(key == 'M' && action == GLFW_PRESS){
		camera->ZoomIn(0.5f);
		renderer->ClearLighting();
	}
	if(key == 'N' && action == GLFW_PRESS){
		camera->ZoomIn(-0.5f);
		renderer->ClearLighting();
	}
}

void GLFWCALL MouseButtonCallback(int key, int action)
{
	updateConfig = true;

	if(TwEventMouseButtonGLFW(key, action)) return;
	
	if(key == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS){
			leftMouseButton = true;
	}
	if(key == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_RELEASE){
			leftMouseButton = false;
	}
	if(key == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS){
			rightMouseButton = true;
	}
	if(key == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_RELEASE){
			rightMouseButton = false;
	}
}

void GLFWCALL MousePositionCallback(int x, int y)
{
	if(TwEventMousePosGLFW(x, y)) return;

	float deltaX = x - mousePosition.x;
	float deltaY = y - mousePosition.y;
	mousePosition = glm::vec2(x,y);

	if(leftMouseButton) {
		camera->RotLeft((float)deltaX/windowWidth);
		camera->RotUp((float)deltaY/windowHeight);
		renderer->ClearAccumulationBuffer();
	}
	else if(rightMouseButton) {
		camera->ZoomIn((float)deltaY/windowHeight);
		renderer->ClearAccumulationBuffer();
	}
}

void GLFWCALL MouseWheelCallback(int pos)
{
	if(TwEventMouseWheelGLFW(pos)) return;
}
*/