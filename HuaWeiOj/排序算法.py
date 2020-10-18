# 冒泡排序
a = [35, 12, 99, 18, 76]
res = []
for i in range(len(a)-1):
    for i in range(len(a)-1):
        b = a[i]
        c = a[i+1]
        if b>c :
            a[i] = c
            a[i+1] = b
print("bouble_sort: ",a)
#选择排序
a = [35, 12, 99, 18, 76]
for i in range(len(a)-1):
    for j in range(i+1,len(a)):
        if a[i] > a[j]:
            min = a[j]
            a[j] = a[i]
            a[i] = min
print("select_sort: ",a)
#插入排序算法
'''  c lange
void insert_sort(int arr[], int len)
{
	int i, j, temp;
	for (i = 1; i < len; i++)
	{
		temp = arr[i];
		for (j = i; j > 0 && arr[j - 1] > temp; j--)
		{
			arr[j] = arr[j - 1];
		};
		arr[j] = temp;
		printf("i = %d ", i);
		for (int k = 0; k < len; k++)
		{
			printf("%d ", arr[k]);
		};
		printf("\n");
	}
}
'''