line = input()
lenth = int(input().strip())
para = []
n = 0
for i in line:
    if u'\u4e00'<= i <=u'\u9fff':
        n = n+2
    else:
        n = n+1
    if n>lenth:
        break
    else:
        para.append(i)
print(''.join(para))