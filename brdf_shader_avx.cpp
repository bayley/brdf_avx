#include <math.h>
#include <immintrin.h> 

#include "brdf.h"
#include "geom.h"

__attribute__((noinline))
void PrincipledBRDF::sample_8(vec3f8 * result, float * ci, float * co, float * c_th, float * c_td) {
	__m256 m_one, m_1_over_pi, m_125_over_pi, m_half;
	__m256 m_ci, m_co, m_c_th, m_c_td;
	__m256 m_spec_alpha, m_cc_alpha;
	__m256 m_f_ci, m_f_co, m_f_c_td;

	__m256 m_p_0, m_p_1; //temp parameter storage
	__m256 m_t_0, m_t_1, m_t_2, m_t_3, m_t_4; //intermediate results
	__m256 m_colored, m_white; //result = colored * base_color + white * {1, 1, 1}

	//load constants
	m_one = _mm256_set1_ps(1.f);
	m_1_over_pi = _mm256_set1_ps(1.f / (float) M_PI);
	m_125_over_pi = _mm256_set1_ps(1.25f / (float) M_PI);
	m_half = _mm256_set1_ps(.5f);

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
		m_t_0 = _mm256_set1_ps(0.04f);
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

		m_p_0 = _mm256_set1_ps(0.25f * 0.25f);

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

	m_t_0 = _mm256_broadcast_ss(&base_color.x);
	m_t_0 = _mm256_mul_ps(m_t_0, m_colored);
	m_t_0 = _mm256_add_ps(m_t_0, m_white);
	_mm256_store_ps(result->x, m_t_0);

	m_t_0 = _mm256_broadcast_ss(&base_color.y);
	m_t_0 = _mm256_mul_ps(m_t_0, m_colored);
	m_t_0 = _mm256_add_ps(m_t_0, m_white);
	_mm256_store_ps(result->y, m_t_0);

	m_t_0 = _mm256_broadcast_ss(&base_color.z);
	m_t_0 = _mm256_mul_ps(m_t_0, m_colored);
	m_t_0 = _mm256_add_ps(m_t_0, m_white);
	_mm256_store_ps(result->z, m_t_0);
}
