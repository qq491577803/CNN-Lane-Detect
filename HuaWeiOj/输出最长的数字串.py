data = '023058789abcd12345ed125ss123058789ww'
length = len(data)
i = 0
para = []
while i < length:
    b = []
    a = data[i]
    while data[i].isdigit():
        a = data[i]
        b.append(a)
        i = i + 1
        if i == length:
            break
    if len(b) != 0:
        para.append(b)
    i = i +1
size = [len(i) for i in para]
for i in range(len(size)-1):
    for i in range(len(size) - 1):
        a = size[i]
        b = size[i+1]
        c = para[i]
        d = para[i+1]
        if b>a:
            size[i] = b
            size[i+1] = a
            para[i] = d
            para[i+1] = c
max = size[0]
count = size.count(max)
print(para[count-1])
