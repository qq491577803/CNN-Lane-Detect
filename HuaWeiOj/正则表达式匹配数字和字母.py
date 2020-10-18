import re
data = 'abcd12345ed1-25ss123058789'
a = re.compile('[0-9]*')#匹配0-9数字////注意后面的*有无的却别
b = re.compile('[a-zA-Z]*')#匹配大小写字母
nums = a.findall(data)
alpha = b.findall(data)
print(nums)
print(alpha)
