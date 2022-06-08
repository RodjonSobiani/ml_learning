numbers_list = [2, 4, 7, 3, 14, 19]
for i in numbers_list:
    odd_number = lambda number: number % 2 == 1
    check_list = odd_number(i)
    print(check_list)
