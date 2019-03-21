#include <math.h>

#include "brdf.h"
#include "geom.h"

inline float GTR1(float alpha, float cos) {
  float norm = (alpha * alpha - 1.f) / (M_PI * log(alpha * alpha));
  float tmp = (alpha * alpha - 1.f) * cos * cos + 1.f;
  return norm / tmp;
}

inline float GTR2(float alpha, float cos) {
	float a2 = alpha * alpha;
  float norm = a2 / M_PI;
  float tmp = (a2 - 1.f) * cos * cos + 1.f;
  return norm / (tmp * tmp);
}

inline float GGX(float alpha, float cos) {
  float a = alpha * alpha;
  float b = cos * cos;
  return 1.f / (cos + sqrtf(a + b - a * b));
}

inline float schlick_F(float cos) {
  float u = 1.f - cos;
  float u2 = u * u;
  return u2 * u2 * u;
}

inline float lerp(float a, float b, float t) {
  return a * (1.f - t) + b * t;
}

__attribute__((noinline))
void PrincipledBRDF::sample(vec3f1 * result, float ci, float co, float c_th, float c_td) {
	float colored, white;

	float F_co, F_ci, F_c_td;
  F_co = schlick_F(co);
  F_ci = schlick_F(ci);
  F_c_td = schlick_F(c_td);

  //diffuse
  float fd90 = .5f + 2.f * c_td * c_td * roughness;
  float fd = 1.f / M_PI * lerp(1.f, fd90, F_co) * lerp(1.f, fd90, F_ci);

  float fss90 = c_td * c_td * roughness;
  float fss = lerp(1.f, fss90, F_co) * lerp(1.f, fss90, F_ci);
  float ss = 1.25f / M_PI * (fss * (1.f / (co + ci) - .5f) + .5f);

	colored = lerp(fd, ss, subsurface) * (1.f - metallic);

	//specular D, G
	float alpha = fmaxf(0.001f, roughness * roughness);
  float D_spec = GTR2(alpha, c_th);
  float G_spec = GGX(alpha, ci) * GGX(alpha, co);
	float S_spec = D_spec * G_spec;

	//specular F
	float Fi_spec = specular * 0.08f;
	float Fg_spec = (1 - specular * 0.08f) * F_c_td;
	float Fi_spec_c = Fi_spec * speculartint;
	float Fi_spec_w = Fi_spec * (1 - speculartint);

	//specular = (Fi_c * color + Fi_w * white + Fg * white) * (1 - metallic) + color * metallic
	colored += (Fi_spec_c * (1 - metallic) + metallic) * S_spec;
	white = (Fi_spec_w + Fg_spec) * (1 - metallic) * S_spec;

	//sheen
	float c_sheen = sheen * F_c_td * (1.f - metallic);
	colored += c_sheen * sheentint;
	white += c_sheen * (1 - sheentint);

	//clearcoat
  if (clearcoat > 0.f) {
    float F_cc = lerp(.04f, 1.f, F_c_td);
    float D_cc = GTR1(lerp(.1f, 0.001f, clearcoatgloss), c_th);
    float G_cc = GGX(.25f, ci) * GGX(.25f, co);
    white += F_cc * D_cc * G_cc * clearcoat * 0.25f;
  }

	result->x = base_color.x * colored + white;	
	result->y = base_color.y * colored + white;	
	result->z = base_color.z * colored + white;	
}
