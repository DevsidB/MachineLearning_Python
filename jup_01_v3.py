# pythlist= list(range(10))
# # print (pythlist)
# # print (list(range(20)))
# list1=[]
# for i in range (5):
#     for item in pythlist:
#         list1.append(item*3)
# print (list1)


# pythlist= list(range(10))
# for i in range (5):
#     list1= [item*3 for item in pythlist]
#     print (list1)
# print (list1)


# pythlist= list(range(10))
# for i in range (5):
#     for item in pythlist:
#         list1=[item*3]
#         # print (list1)
# print (list1)


# pythlist= list(range(10))
# for i in range (100000000):
#     list1= [item*3 for item in pythlist]
# print (list1)

pythlist= list(range(1000000))
for i in range (10):
    list1= [item*3 for item in pythlist]
print (list1)