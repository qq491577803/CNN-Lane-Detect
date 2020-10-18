# x = input('Please input a int x =:')
# y = input('Please input a int y =:')
def erToshi(num):
    sum = 0
    for i  in range(len(num)):
        k = len(num) - i-1
        sum = sum + int(num[i]) * 2 **k
    return sum
def shiToer(num):
    temp = [] # 存余数
    last = 0  #保存最后一个商
    while 1:
        temp_ = num // 2  # 熵1
        last_ = num % 2  # 余1
        if temp_ < 2:
            print("<2")
            last  = temp_
            temp.append(last_)
            break
        else:
            print(">=2")
            temp.append(last_)
            num = temp_
    sum_ = str(last)
    print(temp,last)
    temp = temp[::-1]
    for  i in temp:
        sum_ = sum_+str(i)
    sum_ = int(sum_)
    return sum_
def dev_shiToer(num):
    list = []
    while 1 :
        shang,yu = divmod(num,2)
        last_s = 0
        if shang  <2 :
            list.append(yu)
            last_s = shang
            print('1')
            break
        else:
            list.append(yu)
            num = shang
            print('2')
    list = list[::-1]
    num_ = str(shang)
    for i in list:
        num_ = num_+ str(i)
if __name__ == '__main__':
    # x_ = erToshi(x)
    # y_ = erToshi(y)
    # sum = x_+y_
    # sum = shiToer(sum)
    # print('The x+y = ',sum)
    dev_shiToer(8)