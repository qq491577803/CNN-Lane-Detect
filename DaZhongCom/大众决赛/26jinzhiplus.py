# x = input()


def ershiliuToshi(x):
    x_=[]
    for i in x:
        x__ = ord(i)-97
        x_.append(x__)
    sum = 0
    for i in range(len(x_)):
        s = int(x_[i]) * 26 ** (len(x_)-1)
        sum = sum + s
    return sum
def shiToershiliu(num):
    list = []
    last = 0
    while 1:
        shang,yu = divmod(num,26)
        if shang<26:
            print('1')
            yu = yu+97
            list.append(yu)
            last = shang

            break
        else:
            print('2')
            list.append(yu)
            num =shang
    list = list[::-1]
    sum_= str(shang)
    for i in list:
        sum_ = sum_+ str(i)
    return sum_
if __name__ == '__main__':
    # x = 'a'
    # y = 'abcdef'
    # shiToershiliu(100)
    num = shiToershiliu(100)
    print(num)