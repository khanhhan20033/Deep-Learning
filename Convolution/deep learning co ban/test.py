int_list = list(map(int, input("Hay nhap danh sach so nguyen (cach nhau boi dau cach): ").split(" ")))
for i in range(len(int_list)):
    if int_list[i] > 0:
        int_list[i] = 0
    elif int_list[i] < 0:
        int_list[i] = 1
    else:
        int_list[i] = 0
print("Danh sach nhan duoc:")
print(" ".join(str(int_list)))
