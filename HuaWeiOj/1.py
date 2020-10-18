# input1 = input()
# input2 = input()
str='guwldvzrsfurobidegiyazkggfpgmyhlrbfjrjerrbnjdenrdxjfmrhtumfdsedkcmthphgavzxlmpcpwbkwsvplhmkbkgkw'
a,b = divmod(len(str),8)
print(len(str),a,b)
for i in range(a):
    print(str[i*8:(i+1)*8])
para = str[a*8::]
if b == 0:
    pass
else:
    para=para+'0'*(8-b)
    print(para)