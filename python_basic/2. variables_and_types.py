myString = 'hello'
myFloat = float(10)
myInt = 20

if myString == "hello":
    print("String: %s" % myString)
if isinstance(myFloat, float) and myFloat == 10.0:
    print("Float: %f" % myFloat)
if isinstance(myInt, int) and myInt == 20:
    print("Integer: %d" % myInt)
