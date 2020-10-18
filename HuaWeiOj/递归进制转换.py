str = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
def changeBase(n,b):
    x,y = divmod(n,b)
    if x>0:
        return changeBase(x,b) + str[y]
    else:
        return str[y]
#十进制 到二进制
def myfun(input):
    a_ = []
    b_ = []
    while True:
        a, b = divmod(input, 2)
        a_.append(a)
        input = a
        b_.append(b)
        if a < 2:
            break
    res = [a_[-1]]
    for i in range(len(b_)):
        c = b_[len(b_) - i - 1]
        res.append(c)
    print(res)
    print(res.count(1))



if __name__ == '__main__':
        # 153158E
       n = 9
       b = 26
       res = changeBase(n, b)
       print(res)


