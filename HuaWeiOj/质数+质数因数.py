#寻找1到101之间的质数
def  ja(n):
    para = []
    if n >=2:
        para.append(2)
    for i in range(2,n):
        j = 2
        while True:
            a = i %j
            j = j+1
            if a ==0:
                break
            if i ==j:
                para.append(i)
    return para
rea = ja(3)
print(rea)


#求一个数的质数因子

def myfun(n):
    i = 2
    s = n
    para = []
    while i != s+1:
        a,b = divmod(n,i)
        if b == 0:
            n = a
            para.append(i)
        else:
            i = i + 1
    return para

#判断某一个数是否为素数
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

print('flag:',rec(97))