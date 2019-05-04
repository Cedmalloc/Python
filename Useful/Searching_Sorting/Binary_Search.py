def binary_search(input_array, value):
    l = 0
    h = len(input_array)-1
    while l <= h:
        m = (l+h)//2
        if(input_array[m]==value):
            return m
        elif(input_array[m] < value):
            l = m+1
        else:
            h = m-1
    return -1

test_list = [1,3,9,11,15,19,29]
test_val1 = 25
test_val2 = 15
print(binary_search(test_list, test_val1))
print(binary_search(test_list, test_val2))
