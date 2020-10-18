data = 'b2a2'
para = []
para1 = []
a = 0
for i in range(len(data)):
    obj = data[i]
    if obj.isdigit():
        para1.append(int(obj))
        if a == 0:
            para.append(data[0] + data[a+1:i+1])
        else:
            para.append(data[a+1:i+1])
        a = i
print(para)
print(para1)
for i in range(len(para1)-1):
    for i in range(len(para1)-1):
        a = para1[i]
        b = para1[i+1]
        c = para[i]
        d = para[i+1]
        if a <= b:
            pass
        else:
            para1[i] = b
            para1[i+1] = a
            para[i] = d
            para[i+1] = c
res = ''
for iterm in para:
    s = int(iterm[-1])*(iterm[0:-1])
    res = res + s
print(res)