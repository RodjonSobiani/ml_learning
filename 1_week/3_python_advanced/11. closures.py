def multiplier_of(a):
    def helper(b):
        x = a * b
        print(x)

    return helper


multiplywith5 = multiplier_of(4)
multiplywith5(9)
