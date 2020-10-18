lis1 = ['', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'eleven', 'twelve'
    , 'thirteen', 'fourteen', 'fifteen', 'sixteen', 'seventeen', 'eighteen', 'nineteen']
lis2 = ['', '', 'twenty', 'thirty', 'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety']
def exchange(n):
    a,b = divmod(n,100)
    res = ''
    if a !=0:
        if b !=0:
            res = lis1[a] + ' hundred '+'and '
        else:
            res = lis1[a] + ' hundred'
    if 0<b<20:
        res = res + lis1[b]
    if b>=20:
        c,d = divmod(b,10)
        if d !=0:
            res = res + lis2[c] +" "+ lis1[d]
        else:
            res = res + lis2[c] + lis1[d]
    return res
n  = 100000000
res = ''
a,b = divmod(n,10**6)
if a !=0:
    res = res + exchange(a)+' million '
c,d = divmod(b,10**3)
if c !=0:
    res = res + exchange(c)+ ' thousand '
res = res + exchange(d)
print(res)