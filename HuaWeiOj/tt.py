def rec(n):
    if n ==0 or n ==1:
        flag = 0
    if n>=2:
        for i in range(2,n):
            if n % i ==0:
                flag = 0
                break
            else:
                flag =1
    return flag
n = int(input())
data = [int(i) for i in input().strip().split(' ')]
par = []
for i in range(len(data)-1):
    a = data[i]
    for j in range(i+1,len(data)):
        b = data[j]
        par.append([a,b])
re =[]
for i in par:
    plus= sum(i)
    flag = rec(plus)
    if flag ==1:
        re.append(i)
res =[]
print(re)
print(len(re))
for i in re:
    flag1,flag2 = 1,1
    for j in re :
        if j != i:
            if i[0] in j:
                flag1 = 0
            if i[1] in j:
                flag2 =0
    if flag1 ==0 and flag2 ==0:
        pass
    else:
        res.append(i)
print(res)
print(len(res))