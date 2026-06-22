public class Solution {
    public static long factorial(int n) {
        if (n <= 1) return 1;
        long acc = 1;
        for (int k = 2; k <= n; k++) acc *= k;
        return acc;
    }
}
