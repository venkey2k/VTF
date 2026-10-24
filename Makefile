CXX ?= g++
CXXFLAGS ?= -O3 -march=native -funroll-loops -std=c++17 -fPIC

all: libvtf.so

libvtf.so: vtf_core.cpp vtf.hpp
	$(CXX) $(CXXFLAGS) -shared vtf_core.cpp -o libvtf.so -ldl

test: libvtf.so
	python3 example_quickstart.py

bench: libvtf.so
	python3 benchmark_vtf.py

clean:
	rm -f libvtf.so sample_nifty_sensex_lz4_fast.vtf
