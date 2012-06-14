#ifndef _C_EXPORT_H_
#define _C_EXPORT_H_

class COGLTexture2D;

#include <string>

class CExport
{
public:
	CExport();
	~CExport();

	void ExportPFM(COGLTexture2D* pTexture, std::string strFileName);
private:
};

#endif // _C_EXPORT_H_