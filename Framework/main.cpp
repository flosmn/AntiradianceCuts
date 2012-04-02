#define GLFW_DLL
#include <GL/glew.h>
#include <GL/glfw.h>

#include "AntTweakBar.h"

#include "Defines.h"
#include "Macros.h"
#include "CUtils\GLErrorUtil.h"

#include "Render.h"
#include "Camera.h"

#include "CGLResources\CGLResource.h"
#include "CGLResources\CGLBindLock.h"
#include "CGLResources\CGLTexture2D.h"

#include "CMeshResources\CObjFileLoader.h"
#include "CMeshResources\CMeshGeometry.h"
#include "CMeshResources\CMeshMaterial.h"

#include <iostream>
#include <vector>
#include <map>

using std::cout;
using std::cin;
using std::endl;

int windowWidth, windowHeight;
Renderer *renderer;
Camera* camera = 0;

/* 
	GUI vars
*/
int gNumPaths = 1000000;

bool updateConfig = false;
void UpdateConfig();
void CleanUp();
void Quit(int status);
void Init();
void Release();
void GLFWCALL WindowResize(int width, int height);
int GLFWCALL WindowClose(void);
void GLFWCALL KeyCallback(int key, int action);
void GLFWCALL MouseButtonCallback(int key, int action);
void GLFWCALL MousePositionCallback(int x, int y);
void GLFWCALL MouseWheelCallback(int pos);

void test();

glm::vec2 mousePosition;
bool leftMouseButton = false;
bool rightMouseButton = false;

bool config_use_antiradiance = true;
float config_blur_factor = 0.3f;

int main()
{
	srand ( 0 );

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
	if(glfwOpenWindow(windowWidth = 1280, windowHeight = 720, mode.RedBits, mode.GreenBits, mode.BlueBits, 0, 32, 32, GLFW_WINDOW) == GL_FALSE) {
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

	TwBar *myBar;
	myBar = TwNewBar("GUI");
		
	TwAddVarRW(myBar, "Use Antiradiance", TW_TYPE_BOOL32, &config_use_antiradiance, "");
	TwAddVarRW(myBar, "Blur factor", TW_TYPE_FLOAT, &config_blur_factor, " min=0.01 max=2.0 step=0.01 ");

	camera = new Camera(windowWidth, windowHeight, 0.1f, 100.0f); 
	
	renderer = new Renderer(camera);
	if(!renderer->Init()) {
		cout << "renderer initialization failed." << endl;
		Quit(EXIT_FAILURE);
	}

	while(running) {
		if(updateConfig) UpdateConfig();
		
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

	Release();

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

	if(key == 'T' && action == GLFW_PRESS){
		renderer->SetDrawTexture(!renderer->GetDrawTexture());
	}
	
	if(key == 'P' && action == GLFW_PRESS){
		renderer->PrintCameraConfig();
	}

	if(key == 'C' && action == GLFW_PRESS){
		renderer->ClearLighting();
	}

	if(key == 'S' && action == GLFW_PRESS){
		renderer->Stats();
	}

	if(key == 'L' && action == GLFW_PRESS){
		renderer->SetDrawLight(!renderer->GetDrawLight());
	}

	if(key == 'D' && action == GLFW_PRESS){
		renderer->DrawOnlyDirectLight(!renderer->GetDrawOnlyDirectLight());
	}
	
	if(key == 'I' && action == GLFW_PRESS){
		renderer->DrawOnlyIndirectLight(!renderer->GetDrawOnlyIndirectLight());
	}

	if(key == 'A' && action == GLFW_PRESS){
		renderer->UseAntiradiance(!renderer->GetUseAntiradiance());
	}

	if(key == 'B' && action == GLFW_PRESS){
		renderer->DrawAntiradiance(!renderer->GetDrawAntiradiance());
	}

	if(key == 'N' && action == GLFW_PRESS){
		renderer->SetBlurFactor(renderer->GetBlurFactor() * 0.5f);
	}

	if(key == 'M' && action == GLFW_PRESS){
		renderer->SetBlurFactor(renderer->GetBlurFactor() * 2.0f);
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

void UpdateConfig()
{
	renderer->UseAntiradiance(config_use_antiradiance);
	renderer->SetBlurFactor(config_blur_factor);

	updateConfig = false;
}