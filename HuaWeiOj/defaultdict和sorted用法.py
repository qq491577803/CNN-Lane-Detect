#defaultdict 键值可以重复，
# sorted（a,reverse = True） ,sorted(a,key=ord)
#对dd添加键和值 dd[data0.count(i)].append(i)
from collections import defaultdict
data0 = 'aadddccddc'
data = set(data0)
dd = defaultdict(list)
for i in data:
    dd[data0.count(i)].append(i)
res = ''
for i in sorted(dd.keys(),reverse=True):
    res = res + ''.join((sorted(dd[i],key=ord)))
print(res)

from collections import defaultdict
while 1:
    try:
        data = list(input())
        l = list(set(data))
        dic = defaultdict(list)
        for i in l:
            dic[data.count(i)].append(i)
        result = ''
        for key in sorted(dic.keys(),reverse = True):
            result += ''.join(sorted(dic[key],key = ord))
        print(result)
    except:
        break