CXX=clang++-6.0
CPPFLAGS=-O3 -I. -mavx
DEPS = brdf.h geom.h
OBJ = main.o brdf.o brdf_shader.o brdf_shader_avx.o
LIBS = -lm

%.o: %.c $(DEPS)
	$(CXX) -c -o $@ $< $(CPPFLAGS)

brdf_test: $(OBJ)
	$(CXX) -o $@ $^ $(CPPFLAGS) $(LIBS)

.PHONY: clean

clean:
	rm -f *.o brdf_test
