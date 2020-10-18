x = input('input x:')
y = input('input y:')
x_ = ''
y_ = ''
for i in x :
    num  = ord(i)-97
    x_= x_+ str(num)
for j in y:
    num1 = ord(j)-97
    y_ = y_ + str(num1)
x_ = int(x_)
y_ = int(y_)
# if len(str(x_)!=1) and str(x_)[0] == 'a' or str(y_)[0] == 'a':
#     raise ValueError('Input error ...')

print('x_,y_:',x_,y_)
sum_x = 0
if x_ > 25:
    for  i in range(len(str(x_))):
        sum_x =  sum_x + int(str(x_)[i])*26**(len(str(x_))-i-1)
else:
    sum_x = x_
sum_y = 0
if y_ >25:
    for  i in range(len(str(y_))):
        sum_y =  sum_y + int(str(y_)[i])*26**(len(str(y_))-i-1)
else:
    sum_y = y_
print(sum_y,sum_x)
sum_x_y = sum_x+sum_y
print('The result is :',sum_x_y)


# input1 = input('input1:')
# input2 = input('input2:')
# for  i in range(len(input2)):
#     str = input2[i]
#     if str in input1:
#         flag = 1
#     else:
#         flag  = 0
#         break
# if flag == 1:
#     print('True')
# else:
#     print('False')
#
