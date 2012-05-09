#ifndef _C_EXPORT_H_
#define _C_EXPORT_H_

class CGLTexture2D;

#include <string>

class CExport
{
public:
	CExport();
	~CExport();

	void ExportPFM(CGLTexture2D* pTexture, std::string strFileName);
private:
};

#endif // _C_EXPORT_H_