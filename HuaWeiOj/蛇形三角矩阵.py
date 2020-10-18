def a(n):
    if n ==1:
        return 1
    return a(n-1) + (n-1)
def b(i,j):
    delta = j+i-1
    if j == 1:
        return a(i)
    return  b(i,j-1) + delta
n = int(input())
for i in range(1,n+1):
    a0 = a(i)#每一行第一个元素
    ns = n - i +1#每一行一共有多少个元素
    for j in range(1,ns+1):
        print(b(i,j),end= '')
        print(' ',end='')
    print()