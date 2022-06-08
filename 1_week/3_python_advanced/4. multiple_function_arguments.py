# edit the functions prototype and implementation
def foo(a, b, c, *other):
    if len(other) == 1:
        return 1
    if len(other) == 2:
        return 2


def bar(a, b, c, **options):
    if options.get("magicnumber") == 6:
        return False
    if options.get("magicnumber") == 7:
        return True


# test code
if foo(1, 2, 3, 4) == 1:
    print("Good.")
if foo(1, 2, 3, 4, 5) == 2:
    print("Better.")
if not bar(1, 2, 3, magicnumber=6):
    print("Great.")
if bar(1, 2, 3, magicnumber=7):
    print("Awesome!")
