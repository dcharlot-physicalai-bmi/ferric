// Stub of ggml.h for json.cpp: GGML_ASSERT only.
#pragma once
#include <cstdio>
#include <cstdlib>
#define GGML_ASSERT(x) do { if (!(x)) { fprintf(stderr, "GGML_ASSERT(%s) failed\n", #x); abort(); } } while (0)
