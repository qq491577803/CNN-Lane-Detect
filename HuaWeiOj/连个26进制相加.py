str = 'abcdefghijklmnopqrstuvwxyz'
def changeBase(n,b):
    x,y = divmod(n,b)
    if x>0:
        return changeBase(x,b) + str[y]
    else:
        return str[y]

c = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7, 'i': 8, 'j': 9, 'k': 10, 'l': 11, 'm': 12, 'n': 13, 'o': 14, 'p': 15, 'q': 16, 'r': 17, 's': 18, 't': 19, 'u': 20, 'v': 21, 'w': 22, 'x': 23, 'y': 24, 'z': 25}
def rechange(a):
    sum =0
    for i in range(len(a)):
        m = len(a) - i -1
        n = c[a[i]]
        sum = sum + n * 26**(m)
    return sum


if __name__ == '__main__':
    a = 'b'
    b = 'b'
    a =rechange(a)
    b = rechange(b)
    sum = a + b
    sum = changeBase(sum,26)
    print(sum)
