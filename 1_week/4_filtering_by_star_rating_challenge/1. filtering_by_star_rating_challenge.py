# def filter_by_rating(d, rating):
#     print(d, '___________', rating)
#     newDict = dict()
#     # Iterate over all (k,v) pairs in d
#     for key, value in d.items():
#         # Is condition satisfied?
#         if rating(key, value):
#             newDict[key] = value
#     return newDict
#
#
# filter_by_rating({
#     "Luxury Chocolates": "*****",
#     "Tasty Chocolates": "****",
#     "Aunty May Chocolates": "*****",
#     "Generic Chocolates": "***"
# }, "*****")
#
# print(filter_by_rating)

def filter_by_rating(d, rating):
    ''' Filters dictionary d by function f. '''
    newDict = dict()
    for key, value in d.items():
        if rating(key, value):
            newDict[key] = value
    return newDict


d = {
    "Luxury Chocolates": "*****",
    "Tasty Chocolates": "****",
    "Aunty May Chocolates": "*****",
    "Generic Chocolates": "***"
}

print(filter_by_rating(d, lambda k, v: len(v) == 5))
