#include "libferric.h"
#include <stdio.h>
int main(int argc, char** argv) {
    FerricHandle* h = ferric_load(argv[1]);
    if (!h) { printf("load failed\n"); return 1; }
    char* g = ferric_generate(h, "The capital of France is", 8);
    printf("C   GEN : %s\n", g); ferric_free_string(g);
    char* j = ferric_generate_json(h, "Invent a person.",
        "{\"type\":\"object\",\"properties\":{\"name\":{\"type\":\"string\"},\"age\":{\"type\":\"integer\"}}}", 40);
    printf("C   JSON: %s\n", j); ferric_free_string(j);
    ferric_free(h);
    return 0;
}
