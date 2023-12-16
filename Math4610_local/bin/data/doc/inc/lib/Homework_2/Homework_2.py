import math as m
import numpy as np

"""Homework 1"""

#### Question 1
# a. Machine precision value in single precision. float
# Python does not have a 32-bit float
print()
print("Question 1")
print("a. float precision: ")
print(np.finfo(np.float32).eps)

# b. Machine precision value in double precision. double
# Note that Python's float is actually a double (64-bit)
previous_eps = 0.0
eps = 1.0
while (1.0 + eps != 1.0):
    previous_eps = eps
    eps /= 2.0
print("b. double precision: ")
print(previous_eps)
    

### Question 2
#a. return length of a vector
def l_2_nom(vector):
    length = 0
    for i in vector:
        length += i**2
    return m.sqrt(length)

#b. return l-norm
def l_1_norm(vector):
    length = 0
    for i in vector:
        length += abs(i)
    return length

#c. return infinity norm
def infinity_norm(vector):
    max = vector[0]
    for i in vector:
        abs_i = abs(i)
        if max < abs_i:
            max = abs_i
    return max


### Question 3
#a. distance between 2 vectors in l2
def l2_distance(u, v):
    distance = 0
    for i, j in zip(u, v):
        distance += (i - j)**2
        
    return distance

#b. distance between 2 vectors in l1
def l1_distance(u, v):
    distance = 0
    for i, j in zip(u, v):
        distance += abs(i - j)  
    return distance

#c. distance between 2 infinity vectors
def infinity_distance(u, v):
    return abs(infinity_norm(u) - infinity_norm(v))


def report():    
    # print()
    # print("Question 2")
    # print("a. l2 norm: ")
    # print(l_2_nom([1,2,3]))
    # print(l_2_nom([2,-2,2,-2]))
    # print(l_2_nom([0,5,-11,15,33]))
    # print("b. l1 norm: ")
    # print(l_1_norm([1,2,3]))
    # print(l_1_norm([2,-2,2,-2]))
    # print(l_1_norm([0,5,-11,15,33]))
    # print("c. infinity norm: ")
    # print(infinity_norm([3]))
    # print(infinity_norm([1,100,11]))
    # print(infinity_norm([30,-1001,200]))
    # print()
    # print("Question 3")
    # print("a. l2 distance: ")
    # print(l2_distance([3], [4]))
    print(l2_distance([2,2], [-1000,3]))
    print(l1_distance([2,2], [-1000,3]))
    # print(l2_distance([1,5,3], [3,2,1]))
    # print("b. l1 distance: ")
    # print(l1_distance([1], [3]))
    # print(l1_distance([1,2], [3,4]))
    # print(l1_distance([1,5,3], [3,2,1]))
    # print("c. infinity distance: ")
    # print(infinity_distance([1], [3]))
    # print(infinity_distance([1,7], [3,1]))
    # print(infinity_distance([1,2,3], [100, 1000, 10000]))
    
    
### MAIN ###
# Calls the report function to show computations
def main():
    report()

if __name__ == "__main__":
    main()
        
    
    