#include <math.h>
#include <stdio.h>
#include <time.h>

#include <xmmintrin.h>
#include <emmintrin.h>
#include <immintrin.h>

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

class PrincipledBRDF {
public:
  PrincipledBRDF() {}
public:
	void bake();
public:
	float sample(float ci, float co, float c_th, float c_td);
	__m256 sample_8(float * ci, float * co, float * c_th, float * c_td);
public:
  float subsurface, metallic, specular, speculartint, roughness, anisotropic, sheen, sheentint, clearcoat, clearcoatgloss;
private:
	float spec_alpha, cc_alpha, spec_strength, cc_strength, log_cc_alpha2_inv;
};

void PrincipledBRDF::bake() {
	spec_alpha = fmaxf(0.001f, roughness * roughness);
	cc_alpha = 0.1f * (1.f - clearcoatgloss) + 0.001f * clearcoatgloss;
	spec_strength = 0.08f * specular;
	cc_strength = 0.25f * clearcoat;
	log_cc_alpha2_inv = 1.f / log(cc_alpha * cc_alpha);
}

__attribute__((noinline))
float PrincipledBRDF::sample(float ci, float co, float c_th, float c_td) {
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
	
	return white;
}

__attribute__((noinline))
__m256 PrincipledBRDF::sample_8(float * ci, float * co, float * c_th, float * c_td) {
	float b_tmp;
	__m256 m_one, m_1_over_pi, m_125_over_pi, m_half;
	__m256 m_ci, m_co, m_c_th, m_c_td;
	__m256 m_spec_alpha, m_cc_alpha;
	__m256 m_f_ci, m_f_co, m_f_c_td;

	__m256 m_p_0, m_p_1; //temp parameter storage
	__m256 m_t_0, m_t_1, m_t_2, m_t_3, m_t_4; //intermediate results
	__m256 m_colored, m_white; //result = colored * base_color + white * {1, 1, 1}

	//broadcast constants
	b_tmp = 1.f; m_one = _mm256_broadcast_ss(&b_tmp);
	b_tmp = 1.f / (float)M_PI; m_1_over_pi = _mm256_broadcast_ss(&b_tmp);
	b_tmp = 1.25f / (float)M_PI; m_125_over_pi = _mm256_broadcast_ss(&b_tmp);
	b_tmp = .5f; m_half = _mm256_broadcast_ss(&b_tmp);

	//load parameters
	m_ci = _mm256_load_ps(ci);
	m_co = _mm256_load_ps(co);
	m_c_th = _mm256_load_ps(c_th);
	m_c_td = _mm256_load_ps(c_td);

	//broadcast material parameters
	m_spec_alpha = _mm256_broadcast_ss(&spec_alpha);
	m_cc_alpha = _mm256_broadcast_ss(&cc_alpha);

	//calculate fresnel terms
	m_t_0 = _mm256_sub_ps(m_one, m_ci);	
	m_f_ci = _mm256_mul_ps(m_t_0, m_t_0);
	m_f_ci = _mm256_mul_ps(m_f_ci, m_f_ci);
	m_f_ci = _mm256_mul_ps(m_f_ci, m_t_0);

	m_t_0 = _mm256_sub_ps(m_one, m_co);	
	m_f_co = _mm256_mul_ps(m_t_0, m_t_0);
	m_f_co = _mm256_mul_ps(m_f_co, m_f_co);
	m_f_co = _mm256_mul_ps(m_f_co, m_t_0);

	m_t_0 = _mm256_sub_ps(m_one, m_c_td);	
	m_f_c_td = _mm256_mul_ps(m_t_0, m_t_0);
	m_f_c_td = _mm256_mul_ps(m_f_c_td, m_f_c_td);
	m_f_c_td = _mm256_mul_ps(m_f_c_td, m_t_0);

	//diffuse
	m_t_0 = _mm256_mul_ps(m_c_td, m_c_td);
	m_p_0 = _mm256_broadcast_ss(&roughness);
	m_t_0 = _mm256_mul_ps(m_t_0, m_p_0);

	m_t_1 = _mm256_add_ps(m_t_0, m_t_0);
	m_t_1 = _mm256_sub_ps(m_t_1, m_half); //2 * cos(theta_d) * cos(theta_d) * roughness - 0.5

	m_t_2 = _mm256_mul_ps(m_t_1, m_f_co);
	m_t_2 = _mm256_add_ps(m_t_2, m_one);

	m_t_3 = _mm256_mul_ps(m_t_1, m_f_ci);
	m_t_3 = _mm256_add_ps(m_t_3, m_one);

	m_t_2 = _mm256_mul_ps(m_t_2, m_t_3);
	m_t_2 = _mm256_mul_ps(m_t_2, m_1_over_pi);

	m_p_0 = _mm256_broadcast_ss(&subsurface);
	m_t_3 = _mm256_sub_ps(m_one, m_p_0);
	m_t_2 = _mm256_mul_ps(m_t_2, m_t_3); //fd * (1 - subsurface)

	m_t_0 = _mm256_sub_ps(m_t_0, m_one); //cos(theta_d) * cos(theta_d) * roughness - 1

	m_t_1 = _mm256_mul_ps(m_t_0, m_f_co);
	m_t_1 = _mm256_add_ps(m_t_1, m_one);

	m_t_3 = _mm256_mul_ps(m_t_0, m_f_ci);
	m_t_3 = _mm256_add_ps(m_t_3, m_one);
	
	m_t_1 = _mm256_mul_ps(m_t_1, m_t_3); //fss

	m_t_3 = _mm256_add_ps(m_ci, m_co);
	m_t_3 = _mm256_div_ps(m_one, m_t_3);
	m_t_3 = _mm256_sub_ps(m_t_3, m_half);
	m_t_1 = _mm256_mul_ps(m_t_1, m_t_3);
	m_t_1 = _mm256_add_ps(m_t_1, m_half);
	m_t_1 = _mm256_mul_ps(m_125_over_pi, m_t_1);

	m_t_1 = _mm256_mul_ps(m_t_1, m_p_0); //ss * subsurface	

	m_colored = _mm256_add_ps(m_t_1, m_t_2);
	m_p_0 = _mm256_broadcast_ss(&metallic);
	m_p_0 = _mm256_sub_ps(m_one, m_p_0);
	m_colored = _mm256_mul_ps(m_colored, m_p_0); //diffuse

	//specular D, G
	m_t_0 = _mm256_mul_ps(m_spec_alpha, m_spec_alpha);
	m_t_1 = _mm256_sub_ps(m_t_0, m_one);
	m_t_1 = _mm256_mul_ps(m_t_1, m_c_th);
	m_t_1 = _mm256_mul_ps(m_t_1, m_c_th);
	m_t_1 = _mm256_add_ps(m_t_1, m_one);
	m_t_1 = _mm256_mul_ps(m_t_1, m_t_1);
	m_t_1 = _mm256_div_ps(m_t_0, m_t_1);
	m_t_1 = _mm256_mul_ps(m_t_1, m_1_over_pi); //GTR2(roughness^2, cos(theta_h))	

	m_t_2 = _mm256_mul_ps(m_ci, m_ci);
	m_t_3 = _mm256_mul_ps(m_t_2, m_t_0);
	m_t_3 = _mm256_sub_ps(m_t_0, m_t_3);
	m_t_3 = _mm256_add_ps(m_t_3, m_t_2);
	m_t_3 = _mm256_sqrt_ps(m_t_3);
	m_t_3 = _mm256_add_ps(m_ci, m_t_3); //1 / GGX(roughness^2, cos(theta_i)) 

	m_t_2 = _mm256_mul_ps(m_co, m_co);
	m_t_4 = _mm256_mul_ps(m_t_2, m_t_0);
	m_t_4 = _mm256_sub_ps(m_t_0, m_t_4);
	m_t_4 = _mm256_add_ps(m_t_4, m_t_2);
	m_t_4 = _mm256_sqrt_ps(m_t_4);
	m_t_4 = _mm256_add_ps(m_co, m_t_4); //1 / GGX(roughness^2, cos(theta_o))

	m_t_4 = _mm256_mul_ps(m_t_3, m_t_4);
	m_t_4 = _mm256_div_ps(m_t_1, m_t_4); //GTR2 * GGX * GGX

	//colored, white parts of specular
	m_p_1 = _mm256_broadcast_ss(&spec_strength);
	m_t_0 = _mm256_mul_ps(m_p_0, m_p_1); //0.08 * specular * (1 - metallic)
	
	m_t_2 = _mm256_broadcast_ss(&speculartint); //speculartint
	m_t_3 = _mm256_sub_ps(m_one, m_t_2); //1 - speculartint

	m_t_2 = _mm256_mul_ps(m_t_0, m_t_2); //0.08 * specular * (1 - metallic) * speculartint
	m_t_3 = _mm256_mul_ps(m_t_0, m_t_3); //0.08 * specular * (1 - metallic) * (1 - speculartint)
	
	m_p_0 = _mm256_sub_ps(m_one, m_p_0); //metallic
	m_t_2 = _mm256_add_ps(m_t_2, m_p_0); 

	m_t_2 = _mm256_mul_ps(m_t_2, m_t_4);

	m_colored = _mm256_add_ps(m_colored, m_t_2);

	m_p_1 = _mm256_sub_ps(m_one, m_p_1); //grazing
	m_p_1 = _mm256_mul_ps(m_p_1, m_f_c_td);
	m_p_0 = _mm256_sub_ps(m_one, m_p_0);
	m_p_1 = _mm256_mul_ps(m_p_1, m_p_0);
	
	m_white = _mm256_add_ps(m_t_3, m_p_1);
	m_white = _mm256_mul_ps(m_white, m_t_4); //this is weird and confusing, but it saves a register
	
	//sheen
	m_p_1 = _mm256_broadcast_ss(&sheen);
	m_t_0 = _mm256_mul_ps(m_p_1, m_p_0);
	m_t_0 = _mm256_mul_ps(m_t_0, m_f_c_td);

	m_p_1 = _mm256_broadcast_ss(&sheentint);
	m_t_1 = _mm256_mul_ps(m_t_0, m_p_1);
	
	m_p_1 = _mm256_sub_ps(m_one, m_p_1);
	m_t_2 = _mm256_mul_ps(m_t_0, m_p_1);

	m_colored = _mm256_add_ps(m_colored, m_t_1);
	m_white = _mm256_add_ps(m_white, m_t_2);

	if (clearcoat > 0.f) {
		b_tmp = 0.04f; m_t_0 = _mm256_broadcast_ss(&b_tmp);
		m_t_1 = _mm256_sub_ps(m_one, m_t_0);
		m_t_1 = _mm256_mul_ps(m_t_1, m_f_c_td);
		m_t_0 = _mm256_add_ps(m_t_0, m_t_1); //0.04 + 0.96 * F_c_td

		m_p_0 = _mm256_broadcast_ss(&cc_strength);
		m_t_0 = _mm256_mul_ps(m_p_0, m_t_0); //0.25 * clearcoatstrength * F_cc
	
		m_p_0 = _mm256_broadcast_ss(&cc_alpha);
		m_p_0 = _mm256_mul_ps(m_p_0, m_p_0);

		m_t_1 = _mm256_sub_ps(m_p_0, m_one);
		m_t_1 = _mm256_mul_ps(m_t_1, m_c_th);
		m_t_1 = _mm256_mul_ps(m_t_1, m_c_th);
		m_t_1 = _mm256_add_ps(m_t_1, m_one);

		m_p_0 = _mm256_sub_ps(m_p_0, m_one);
		m_p_0 = _mm256_mul_ps(m_p_0, m_1_over_pi);
		m_p_1 = _mm256_broadcast_ss(&log_cc_alpha2_inv);

		m_p_0 = _mm256_mul_ps(m_p_0, m_p_1);
		m_t_1 = _mm256_div_ps(m_p_0, m_t_1); //GTR1(cc_alpha, cos(theta_h))

		m_t_0 = _mm256_mul_ps(m_t_0, m_t_1);

		b_tmp = 0.25f * 0.25f; m_p_0 = _mm256_broadcast_ss(&b_tmp);

		m_p_1 = _mm256_mul_ps(m_ci, m_ci);
		m_t_1 = _mm256_mul_ps(m_p_0, m_p_1);
		m_t_1 = _mm256_sub_ps(m_p_1, m_t_1);
		m_t_1 = _mm256_add_ps(m_p_0, m_t_1);
		m_t_1 = _mm256_sqrt_ps(m_t_1);
		m_t_1 = _mm256_add_ps(m_ci, m_t_1); //1 / GGX(0.25, cos(theta_i))

		m_p_1 = _mm256_mul_ps(m_co, m_co);
		m_t_2 = _mm256_mul_ps(m_p_0, m_p_1);
		m_t_2 = _mm256_sub_ps(m_p_1, m_t_2);
		m_t_2 = _mm256_add_ps(m_p_0, m_t_2);
		m_t_2 = _mm256_sqrt_ps(m_t_2);
		m_t_2 = _mm256_add_ps(m_co, m_t_2); //1 / GGX(0.25, cos(theta_o))

		m_t_1 = _mm256_mul_ps(m_t_1, m_t_2);
		m_t_0 = _mm256_div_ps(m_t_0, m_t_1);

		m_white = _mm256_add_ps(m_white, m_t_0);
	}

	return m_white;
}

float randf() {
	return (float)((double)rand() / (double) RAND_MAX);
}

int main(int argc, char ** argv) {
	srand(time(0));

	float ci[8] __attribute__ ((aligned (32)));
	float co[8] __attribute__ ((aligned (32)));
	float c_th[8] __attribute__ ((aligned (32)));
	float c_td[8] __attribute__ ((aligned (32)));

	PrincipledBRDF m;
	m.subsurface = 0.3f;
	m.metallic = 0.2f;
	m.specular = 0.6f;
	m.speculartint = 0.4f;
	m.roughness = 0.7f;
	m.anisotropic = 0.f;
	m.sheen = 0.35f;
	m.sheentint = 0.75f;
	m.clearcoat = 0.1f;
	m.clearcoatgloss = 0.4f;
	m.bake();

	//check that the results are equal
	printf("Scalar results: ");
	for (int i = 0; i < 8; i++) {
		ci[i] = randf();
		co[i] = randf();
		c_th[i] = randf();
		c_td[i] = randf();
		printf("%f ", m.sample(ci[i], co[i], c_th[i], c_td[i]));
	}
	printf("\n");

	union {__m256 v; float d[8] __attribute__ ((aligned (32)));};

	printf("Vector results: ");
	v = m.sample_8(ci, co, c_th, c_td);
	for (int i = 0; i < 8; i++) {
		printf("%f ", d[i]);
	}
	printf("\n");

	//execution timing
	clock_t start, end;
	double t_scalar, t_vector;

	volatile float tmp_f;
	volatile __m256 tmp_v;

	int iters = 1000000;
	
	start = clock();
	for (int i = 0; i < iters; i++) {
		for (int j = 0; j < 8; j++) {
			tmp_f = m.sample(ci[j], co[j], c_th[j], c_td[j]);
		}
	}
	end = clock();
	t_scalar = (double)(end - start) / CLOCKS_PER_SEC;

	start = clock();
	for (int i = 0; i < iters; i++) {
		tmp_v = m.sample_8(ci, co, c_th, c_td);
	}
	end = clock();
	t_vector = (double)(end - start) / CLOCKS_PER_SEC;

	printf("Scalar: %f s || Vector %f s\n", t_scalar, t_vector);
	printf("Speedup: %f x\n", t_scalar / t_vector);
}
