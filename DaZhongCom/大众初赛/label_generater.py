N=1
with open('./LightExmple/label.txt', 'w') as fn:
    for i in range(523):
        if N<150:
            fn.write('1'+'\n')
            print("1:",N)
        if 149<N<300:
            fn.write('2'+'\n')
            print("2:",N)
        if 299<N<416:
            fn.write('3'+'\n')
            print("3",N)
        if 415 < N < 524:
            fn.write('4' + '\n')
            print("4:",N)
        N=N+1