a = ['a','b','c']

for idx, elem in enumerate(a):
    print(idx, elem)

for idx, elem in reversed(list(enumerate(a))):
    print(idx, elem)