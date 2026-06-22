#include <stdint.h>
int32_t overflow_add(int32_t a, int32_t b) {
    return (int32_t)((int64_t)a + (int64_t)b);
}
