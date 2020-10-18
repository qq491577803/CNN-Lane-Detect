input1 = int(input())
input2 = []
for i in range(input1):
    a = input()
    input2.append(a)
for iterm in input2:
    a,b = divmod(len(iterm),8)
    for i in range(a):
        print(iterm[(i*8):((i+1)*8)])
    if b !=0:
        print(iterm[a*8:len(iterm)]+(8-b)*'0')