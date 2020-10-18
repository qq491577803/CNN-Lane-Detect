data = 'abcd12#%XYZ'
res = ''
for i in data:
    if i.isalpha():
        if i.isupper():
            res = res + i.lower()
        if i.islower():
            res = res + i.upper()
    else:
        res =res + i
        i = i
print(res)