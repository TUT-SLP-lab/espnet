file1 = 'tokens_APS+SPS.txt'
file2 = 'tokens_jnas.txt'
# file3 = 'tokens_jnas.txt'

with open(file1) as f1:
    with open(file2) as f2:
        list1 = [s.strip('\n') for s in f1.readlines()]
        list2 = [s.strip('\n') for s in f2.readlines()]

# count = 0
# for i in list1:
#     for j in list2:
#         if i == j:
#             count += 1
# print(count)

print(file1, ':', len(list1))
print(file2, ':', len(list2))
# print(file3, ':', len(list3))
union = set(list1) & set(list2)
# diff = set(list1) ^ set(list2)
print('match :', len(union))
# print('diff :', len(diff))