#ifndef __BRDF_H
#define __BRDF_H

#include <immintrin.h>
#include "geom.h"

class PrincipledBRDF {
public:
  PrincipledBRDF() {}
public:
	void bake();
public:
	void sample(vec3f1 * result, float ci, float co, float c_th, float c_td);
	void sample_8(vec3f8 * result, float * ci, float * co, float * c_th, float * c_td);
public:
  float subsurface, metallic, specular, speculartint, roughness, anisotropic, sheen, sheentint, clearcoat, clearcoatgloss;
	vec3f1 base_color;
private:
	float spec_alpha, cc_alpha, spec_strength, cc_strength, log_cc_alpha2_inv;
};

#endif

