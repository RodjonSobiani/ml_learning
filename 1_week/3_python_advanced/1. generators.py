import types


def fib():
    a = 0
    b = 1
    while True:
        c = b + a
        b = a
        a = c
        yield c


if type(fib()) == types.GeneratorType:
    print("Good, The fib function is a generator.")

    counter = 0
    for n in fib():
        print(n, end=' ')
        counter += 1
        if counter == 10:
            break
