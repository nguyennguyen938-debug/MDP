#include <stdint.h>
int64_t array_max(int64_t *arr, int32_t n) {
    if (n <= 0) return 0;
    int64_t best = arr[0];
    for (int32_t i = 1; i < n; i++) if (arr[i] > best) best = arr[i];
    return best;
}
