data = input()
data = data.split(' ')
a = int(data[0])
b = int(data[1])
if a>=b:
    res = b
else:
    res =a
while res<=(a*b) :
    c,d = divmod(res,a)
    e,f = divmod(res,b)
    if d ==0 and f==0:
        print(res)
        break
    res =res +1