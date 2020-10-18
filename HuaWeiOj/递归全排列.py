COUNT=0
def perm(n,begin,end):
    global COUNT
    if begin>=end:
        COUNT +=1
        print(n)
    else:
        i=begin
        for num in range(begin,end):
            n[num],n[i]=n[i],n[num]
            perm(n,begin+1,end)
            n[num],n[i]=n[i],n[num]

n=['a','a','c','d','e','f','g']
perm(n,0,len(n))
print(COUNT)