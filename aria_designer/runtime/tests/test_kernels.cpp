#include "kernels.h"
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>

#define EPS 1e-5

void test_relu() {
    printf("Testing ReLU... ");
    float x[] = {-1.0f, 0.0f, 1.0f, 2.0f};
    float y[4];
    aria_relu_f32(x, y, 4);
    assert(y[0] == 0.0f);
    assert(y[1] == 0.0f);
    assert(y[2] == 1.0f);
    assert(y[3] == 2.0f);
    printf("OK\n");
}

void test_add() {
    printf("Testing Add... ");
    float a[] = {1.0f, 2.0f, 3.0f};
    float b[] = {4.0f, 5.0f, 6.0f};
    float y[3];
    aria_add_f32(a, b, y, 3);
    assert(y[0] == 5.0f);
    assert(y[1] == 7.0f);
    assert(y[2] == 9.0f);
    printf("OK\n");
}

void test_matmul() {
    printf("Testing MatMul... ");
    /* 2x3 @ 3x2 -> 2x2
     * [1, 2, 3]   [7,  8]    [58,  64]
     * [4, 5, 6] @ [9, 10] = [139, 154]
     *             [11, 12]
     */
    float A[] = {1, 2, 3, 4, 5, 6};
    float B[] = {7, 8, 9, 10, 11, 12};
    float C[4];
    aria_matmul_f32(A, B, C, 2, 3, 2);
    assert(fabs(C[0] - 58.0f) < EPS);
    assert(fabs(C[1] - 64.0f) < EPS);
    assert(fabs(C[2] - 139.0f) < EPS);
    assert(fabs(C[3] - 154.0f) < EPS);
    printf("OK\n");
}

void test_tropical_add() {
    printf("Testing Tropical Add... ");
    float a[] = {1.0f, 2.0f, 3.0f};
    float b[] = {3.0f, 1.0f, 4.0f};
    float y[3];
    aria_tropical_add_f32(a, b, y, 3);
    assert(y[0] == 1.0f);
    assert(y[1] == 1.0f);
    assert(y[2] == 3.0f);
    printf("OK\n");
}

void test_tropical_matmul() {
    printf("Testing Tropical MatMul... ");
    float A[] = {1, 2, 3, 4};
    float B[] = {5, 6, 7, 8};
    float C[4];
    aria_tropical_matmul_f32(A, B, C, 2, 2, 2);
    assert(fabs(C[0] - 6.0f) < EPS);
    assert(fabs(C[1] - 7.0f) < EPS);
    assert(fabs(C[2] - 8.0f) < EPS);
    assert(fabs(C[3] - 9.0f) < EPS);
    printf("OK\n");
}

void test_tropical_center() {
    printf("Testing Tropical Center... ");
    /* x shape: (1, 3, 2) — batch=1, seq=3, dim=2
     * Kernel does running-min centering per dimension:
     *   y[s] = x[s] - min(x[0..s])
     *
     * dim=0: [5, 3, 7] → running_min=[5,3,3] → [0, 0, 4]
     * dim=1: [1, 4, 0] → running_min=[1,1,0] → [0, 3, 0]
     * Interleaved: [0, 0, 0, 3, 4, 0]
     */
    float x[] = {
        5.0f, 1.0f,
        3.0f, 4.0f,
        7.0f, 0.0f
    };
    float y[6];
    aria_tropical_center_f32(x, y, 1, 3, 2);
    assert(fabs(y[0] - 0.0f) < EPS);
    assert(fabs(y[1] - 0.0f) < EPS);
    assert(fabs(y[2] - 0.0f) < EPS);
    assert(fabs(y[3] - 3.0f) < EPS);
    assert(fabs(y[4] - 4.0f) < EPS);
    assert(fabs(y[5] - 0.0f) < EPS);
    printf("OK\n");
}

void test_hyp_distance() {
    printf("Testing Hyperbolic Distance... ");
    float x[] = {0.1f, 0.0f, 0.0f, 0.2f};
    float y[] = {0.1f, 0.0f, 0.0f, 0.2f};
    float out[2];
    aria_hyp_distance_f32(x, y, out, 1, 2, 2);
    assert(fabs(out[0]) < 1e-3f);
    assert(fabs(out[1]) < 1e-3f);
    printf("OK\n");
}

void test_padic_gate() {
    printf("Testing P-adic Gate... ");
    float x[] = {1.0f, 0.5f, 0.0f, -0.25f};
    float y[4];
    aria_padic_gate_f32(x, y, 4, 2.0f);
    assert(fabs(y[0] - 0.5f) < 1e-4f);
    assert(fabs(y[2]) < 1e-6f);
    assert(y[1] > 0.0f);
    assert(y[3] < 0.0f);
    printf("OK\n");
}

void test_tropical_attention() {
    printf("Testing Tropical Attention... ");
    float x[] = {
        1.0f, 2.0f,
        1.0f, 2.0f
    };
    float y[4];
    aria_tropical_attention_f32(x, y, 1, 2, 2, 0.1f);
    assert(fabs(y[0] - 1.0f) < 1e-4f);
    assert(fabs(y[1] - 2.0f) < 1e-4f);
    assert(fabs(y[2] - 1.0f) < 1e-4f);
    assert(fabs(y[3] - 2.0f) < 1e-4f);
    printf("OK\n");
}

void test_tropical_gate() {
    printf("Testing Tropical Gate... ");
    float x[] = {
        1.0f, 2.0f,
        1.0f, 2.0f
    };
    float y[4];
    aria_tropical_gate_f32(x, y, 1, 2, 2, 0.1f);
    float e0 = 1.0f / (1.0f + expf(-1.0f));
    float e1 = 1.0f / (1.0f + expf(-2.0f));
    assert(fabs(y[0] - 1.0f * e0) < 1e-4f);
    assert(fabs(y[1] - 2.0f * e1) < 1e-4f);
    assert(fabs(y[2] - 1.0f * e0) < 1e-4f);
    assert(fabs(y[3] - 2.0f * e1) < 1e-4f);
    printf("OK\n");
}

void test_rmsnorm() {
    printf("Testing RMSNorm... ");
    float x[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float w[] = {1.0f, 1.0f, 1.0f, 1.0f};
    float y[4];
    /* rms = sqrt((1^2+2^2+3^2+4^2)/4) = sqrt(30/4) = sqrt(7.5) = 2.73861 */
    aria_rmsnorm_f32(x, w, y, 1, 4, 1e-6f);
    float rms = sqrtf(7.5f);
    assert(fabs(y[0] - 1.0f/rms) < EPS);
    assert(fabs(y[3] - 4.0f/rms) < EPS);
    printf("OK\n");
}

void test_file_loader_csv() {
    printf("Testing File Loader CSV... ");
    const char *path = "/tmp/aria_test_loader.csv";
    FILE *fp = fopen(path, "w");
    assert(fp != NULL);
    fprintf(fp, "a,b,c\n");
    fprintf(fp, "1,2,3\n");
    fprintf(fp, "4,5,6\n");
    fclose(fp);

    float out[16] = {0};
    int rows = aria_file_loader_csv_f32(path, out, 4, 3, ',', 1);
    assert(rows == 2);
    assert(fabs(out[0] - 1.0f) < EPS);
    assert(fabs(out[1] - 2.0f) < EPS);
    assert(fabs(out[2] - 3.0f) < EPS);
    assert(fabs(out[3] - 4.0f) < EPS);
    assert(fabs(out[4] - 5.0f) < EPS);
    assert(fabs(out[5] - 6.0f) < EPS);

    remove(path);
    printf("OK\n");
}

void test_binary_file_reader() {
    printf("Testing Binary File Reader... ");
    const char *path = "/tmp/aria_test_reader.bin";
    FILE *fp = fopen(path, "wb");
    assert(fp != NULL);
    float src[] = {1.5f, -2.0f, 3.25f};
    size_t written = fwrite(src, sizeof(float), 3, fp);
    fclose(fp);
    assert(written == 3);

    float out[8] = {0};
    int n = aria_binary_file_reader_f32(path, out, 8, 0);
    assert(n == 3);
    assert(fabs(out[0] - 1.5f) < EPS);
    assert(fabs(out[1] + 2.0f) < EPS);
    assert(fabs(out[2] - 3.25f) < EPS);

    remove(path);
    printf("OK\n");
}

void test_file_writer_txt() {
    printf("Testing File Writer TXT... ");
    const char *path = "/tmp/aria_test_writer.txt";
    remove(path);

    float src[] = {0.5f, 1.5f, 2.5f};
    int n = aria_file_writer_txt_f32(path, src, 3, 0);
    assert(n == 3);

    FILE *fp = fopen(path, "r");
    assert(fp != NULL);
    char line[64];
    int lines = 0;
    while (fgets(line, sizeof(line), fp)) lines++;
    fclose(fp);
    assert(lines == 3);

    int rc_no_overwrite = aria_file_writer_txt_f32(path, src, 3, 0);
    assert(rc_no_overwrite < 0);

    int rc_overwrite = aria_file_writer_txt_f32(path, src, 3, 1);
    assert(rc_overwrite == 3);

    remove(path);
    printf("OK\n");
}

int main() {
    test_relu();
    test_add();
    test_matmul();
    test_tropical_add();
    test_tropical_matmul();
    test_tropical_center();
    test_hyp_distance();
    test_padic_gate();
    test_tropical_attention();
    test_tropical_gate();
    test_rmsnorm();
    test_file_loader_csv();
    test_binary_file_reader();
    test_file_writer_txt();
    printf("All kernel tests passed!\n");
    return 0;
}
