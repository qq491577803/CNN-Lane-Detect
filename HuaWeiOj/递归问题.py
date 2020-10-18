#a(n) = a(n-3) + a(n-1)
#2 3 4 6 9 13 19
def a(n):
    if n ==1:
        return 2
    if n==2:
        return 3
    if n==3:
        return 4
    return a(n-3)+a(n-1)
for i in range(1,11):
    res = a(i)
    print(res)
res = a(6)
print(res)