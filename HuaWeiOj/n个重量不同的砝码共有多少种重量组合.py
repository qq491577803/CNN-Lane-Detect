nums = ['1', '2', '3', '4', '5']
weight = ['2', '3', '4', '5', '6']
nums = [int(i) for i in nums]
weight = [int(j) for j in weight]
n = 5

res = set()
for i in range(nums[0] + 1):
    res.add(i * weight[0])
for i in range(1, n):
    tmp = list(res)
    for j in range(1, nums[i] + 1):
        for wt in tmp:  # 变成list在这里才能遍历
            res.add(wt + j * weight[i])
print("-----------------")

