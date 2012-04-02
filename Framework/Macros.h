#ifndef _MACROS_H_
#define _MACROS_H_

#define V_RET_FOF(x) if(!(x)) return false;
#define V_RET_FOT(x) if((x)) return false;
#define V_RET_TOF(x) if(!(x)) return true;
#define V_RET_TOT(x) if((x)) return true;

#define SAFE_DELETE(x) if((x)!=NULL) delete (x); (x)=NULL;
#define SAFE_DELETE_ARRAY(x) if((x)!=NULL) delete[] (x); (x)=NULL;

#endif // _MACROS_H_