#include <stdio.h>
#include <time.h>

#include <immintrin.h>

#include "brdf.h"
#include "geom.h"

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
	m.base_color = {0.3f, 0.4f, 0.5f};
	m.bake();

	vec3f v;
	vec3f8 v8;

	//check that the results are equal
	printf("Scalar results:\n");
	for (int i = 0; i < 8; i++) {
		ci[i] = randf();
		co[i] = randf();
		c_th[i] = randf();
		c_td[i] = randf();
		m.sample(&v, ci[i], co[i], c_th[i], c_td[i]);
		printf("[%f %f %f]\n", v.x, v.y, v.z);
	}
	printf("\n");

	printf("Vector results:\n");
	m.sample_8(&v8, ci, co, c_th, c_td);
	for (int i = 0; i < 8; i++) {
		printf("[%f %f %f]\n", v8.x[i], v8.y[i], v8.z[i]);
	}
	printf("\n");

	//execution timing
	clock_t start, end;
	double t_scalar, t_vector;

	int iters = 1000000;
	
	start = clock();
	for (int i = 0; i < iters; i++) {
		for (int j = 0; j < 8; j++) {
			m.sample(&v, ci[j], co[j], c_th[j], c_td[j]);
		}
	}
	end = clock();
	t_scalar = (double)(end - start) / CLOCKS_PER_SEC;

	start = clock();
	for (int i = 0; i < iters; i++) {
		m.sample_8(&v8, ci, co, c_th, c_td);
	}
	end = clock();
	t_vector = (double)(end - start) / CLOCKS_PER_SEC;

	printf("Scalar: %f s || Vector %f s\n", t_scalar, t_vector);
	printf("Speedup: %f x\n", t_scalar / t_vector);
}
