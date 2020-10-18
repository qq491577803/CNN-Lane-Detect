import time
data =list('eeefgghhh')
def paixu():
    data =list('eefgghhhe')
    data_aci = [ord(i) for i in data]
    print(data_aci)
    for i in range(len(data_aci)-1):
        for j in range(len(data_aci) - 1):
            a = data_aci[j]
            b = data_aci[j+1]
            if a>b:
                data_aci[j] = b
                data_aci[j+1] = a
    data = [chr(i) for i in data_aci]
    print(data)

# data_aci = [ord(i) for i in data]
p = {}
for key in data:
    p[key] = p.get(key,0)+1
print('ddddddd',p)
res = ''
keys = list(p.keys())
value = p.values()
while True:
    for key in keys:
        if p[key] != 0 :
            res = res + key
            p[key] = p[key] - 1
    value = list(p.values())
    sum = 0
    for i in value:
        sum = sum + int(i)
    if sum ==0:
        break
print(res)