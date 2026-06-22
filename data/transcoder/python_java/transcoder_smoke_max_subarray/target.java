public class Solution {
    public static int max_subarray_sum(int[] nums) {
        if (nums.length == 0) return 0;
        int best = nums[0], cur = nums[0];
        for (int i = 1; i < nums.length; i++) {
            cur = Math.max(nums[i], cur + nums[i]);
            best = Math.max(best, cur);
        }
        return best;
    }
}
