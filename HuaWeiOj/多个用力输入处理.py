while True:
    try:
        n = int(input())
        res = []
        for i in range(n):
            res.append(int(input()))
        for i in sorted(set(res)):
            print(i)
    except:
        break