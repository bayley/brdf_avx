#ifndef __GEOM_H
#define __GEOM_H

typedef struct {
	float x; float y; float z;
} vec3f;

typedef struct {
	float x[8] __attribute((__aligned__(32)));
	float y[8] __attribute((__aligned__(32)));
	float z[8] __attribute((__aligned__(32)));
} vec3f8;

#endif
