#!/usr/bin/env bash
set -euo pipefail
echo "[VTF v1.0] Compiling C++17 SIMD core (libvtf.so)..."
g++ -O3 -march=native -funroll-loops -std=c++17 -shared -fPIC vtf_core.cpp -o libvtf.so -ldl
echo "[VTF v1.0] Running quickstart verification..."
python3 example_quickstart.py
