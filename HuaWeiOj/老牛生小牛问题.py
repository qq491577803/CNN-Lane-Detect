while True:
    try:
        def cal(n):
            if n == None:
                return
            if n <= 3:
                return 1
            return cal(n-1) + cal(n-3)#上一年的牛 + 今年出生的牛
        m = int(input())
        n = int(input())
        num = m * cal(n+3)
        print(num)
    except:
        break
