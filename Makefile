#CXX=clang++-6.0
CXX=g++
CPPFLAGS=-O3 -I. -mavx 
DEPS = 
OBJ = main.o
LIBS = -lm

%.o: %.c $(DEPS)
	$(CXX) -c -o $@ $< $(CPPFLAGS)

brdf_test: $(OBJ)
	$(CXX) -o $@ $^ $(CPPFLAGS) $(LIBS)

.PHONY: clean

clean:
	rm -f *.o brdf_test
