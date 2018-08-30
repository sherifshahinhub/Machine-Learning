# train = []
# file = open("test", "r")
# for line in file:
#     train.append((line, 'politics'))
# print(train)
file = open("test.txt", encoding='utf-8')
var = file.readline()
print(var[50:])