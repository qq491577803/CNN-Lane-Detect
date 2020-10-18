N=1
with open('./LightExmple/label.txt', 'w') as fn:
    for i in range(465):
        if N<160:
            fn.write('1'+'\n')
            print("1:",N)
        if 159<N<292:
            fn.write('2'+'\n')
            print("2:",N)
        if 291<N<446:
            fn.write('3'+'\n')
            print("3",N)
        if 445 < N < 566:
            fn.write('4' + '\n')
            print("4:",N)
        N=N+1