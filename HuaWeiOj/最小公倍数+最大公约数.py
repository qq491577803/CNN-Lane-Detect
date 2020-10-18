a = 7
b = 5
#myfun 是求最大公约数   最小公倍数 = 两数的乘积/最大公约数
def myfun(a,b):
    if a< b:
        a,b = b,a
    while True:
        c = a % b
        if c == 0:
            res = b
            break
        else:
            a = b
            b = c
    return res
res = myfun(a,b)
print(res)