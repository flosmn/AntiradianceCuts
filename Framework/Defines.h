#ifndef DEFINES_H
#define DEFINES_H

#define SUCCESS 1
#define FAILURE 0

#define SAFE_DELETE(x) if((x)!=NULL) delete (x); (x)=NULL;
#define SAFE_DELETE_ARRAY(x) if((x)!=NULL) delete[] (x); (x)=NULL;

#define ARRAY_COUNT( array ) (sizeof( array ) / (sizeof( array[0] ) * (sizeof( array ) != sizeof(void*) || sizeof( array[0] ) <= sizeof(void*))))

#define PI 3.14159265f

#define LOCAL_DIR ".\\"
#define LOCAL_FILE_DIR "data\\"

#endif