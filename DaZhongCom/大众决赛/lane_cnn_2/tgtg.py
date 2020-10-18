def a(a,b):
    # print('a')
    c = a+b
    return c
def b(a,b,c):
    plus = a(b,c)
    print('plus:',plus)

b(a,2,3)