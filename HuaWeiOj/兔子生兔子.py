def a(n):
    if n == 1:
        return 1
    if n == 2:
        return 1
    if n>2:
        return (n-2)+a(n-1)

res = a(10)
print(res)