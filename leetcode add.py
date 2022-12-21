class Solution:
    def twoSum(self, nums, target):
        ls = []
        for i in range(0, len(nums)):
            item = target - nums[i]
            nums[i] = "done"
            print(nums)
            print (item)
            if item in nums:
                if i != nums.index(item):
                    ls.append(i)
                    ls.append(nums.index(item))
                    return ls
class Solution:
    def twoSum(self, nums, target):
        list = []
        for i in range(0, len(nums)):
            item = target - nums[i]
            if item in nums:
                list.append(i)
                list.append(nums.index(item))
                return list
new= Solution()
# print(new.twoSum([2,7,11,15],9))
print(new.twoSum([3,2,4],6))
# print(new.twoSum([3,3],6))