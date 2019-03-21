#include <math.h>

#include "brdf.h"

void PrincipledBRDF::bake() {
	spec_alpha = fmaxf(0.001f, roughness * roughness);
	cc_alpha = 0.1f * (1.f - clearcoatgloss) + 0.001f * clearcoatgloss;
	spec_strength = 0.08f * specular;
	cc_strength = 0.25f * clearcoat;
	log_cc_alpha2_inv = 1.f / log(cc_alpha * cc_alpha);
}
