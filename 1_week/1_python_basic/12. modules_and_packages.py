import re

find_members = []
function_list = dir(re)

for item in function_list:
    if 'find' in item:
        find_members.append(item)

find_members.sort()
print(find_members)
