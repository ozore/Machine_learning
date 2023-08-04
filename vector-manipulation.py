import numpy as np
import time

### Vector creation

a = np.zeros(4)
print(f"np.zeros(4) :   a = {a}, a shape = {a.shape}, a data type = {a.dtype}")
a = np.zeros((4,))
print(f"np.zeros(4,) :  a = {a}, a shape = {a.shape}, a data type = {a.dtype}")
a = np.random.random_sample(4)
print(f"np.random.random_sample(4): a = {a}, a shape = {a.shape}, a data type = {a.dtype}")


# NumPy routines which allocate memory and fill arrays with value but do not accept shape as input argument
a = np.arange(4.)
print(f"np.arange(4.):     a = {a}, a shape = {a.shape}, a data type = {a.dtype}")
a = np.random.rand(4)
print(f"np.random.rand(4): a = {a}, a shape = {a.shape}, a data type = {a.dtype}")

# NumPy routines which allocate memory and fill with user specified values
a = np.array([5,4,3,2])
print(f"np.array([5,4,3,2]):  a = {a},     a shape = {a.shape}, a data type = {a.dtype}")
a = np.array([5.,4,3,2])
print(f"np.array([5.,4,3,2]): a = {a}, a shape = {a.shape}, a data type = {a.dtype}")


### Access values of a vector 

# Vector indexing operations on 1-D vector 
a = np.arange(10)
print(a)

# Access an element
print(f"a.shape: {a.shape} a[2] = {a[2]}, Accessing an element returns a scalar")

# Access the last element, negative indexes count from the end 
print(f"a[-1] = {a[-1]}")

# Index must be within the rqnge of the vector or they will produce an error 
try : 
    c = a[10]
except Exception as e : 
    print("The error message is :")
    print(e)

### Vector slicing 
#  (start:stop:step)

#  vector slicing opeations 
a = np.arange(10)
print(f"a         = {a}")

# access 5 consecutive elements 
c = a[2:7:2]
print("a[2:7:2] = ", c)

# access all elements index 3 and above
c = a[3:];        print("a[3:]    = ", c)

# access all elements below index 3
c = a[:3];        print("a[:3]    = ", c)

# access all elements
c = a[:];         print("a[:]     = ", c)

# Operations on vector 
a = np.array([1,2,3,4])
print(f"a             : {a}") 

# negate elements of a
b = -a 
print(f"b = -a        : {b}")

# sum all elements of a, returns a scalar
b = np.sum(a) 
print(f"b = np.sum(a) : {b}")

b = np.mean(a)
print(f"b = np.mean(a): {b}")

b = a**2
print(f"b = a**2      : {b}")

# Add vector together of the same size
a = np.array([ 1, 2, 3, 4])
b = np.array([-1,-2, 3, 4])
print(f"Binary operators work element wise: {a + b}")

# Add 2 vectors with differents size
#try a mismatched vector operation
c = np.array([1, 2])
try:
    d = a + c
except Exception as e:
    print("The error message you'll see is:")
    print(e)

# Multiply a vector with a scalar 
a = np.array([1, 2, 3, 4])

# multiply a by a scalar
b = 5 * a 
print(f"b = 5 * a : {b}")

v = np.array([1,2,3,4])
c = np.array([-1,4,3,2])

def dot_product(x,y):
    output = 0
    dimension = x.shape[0]
    for i in range(dimension):
        output = output + x[i] *y[i]
    print(f"{output}")
    return output

dot_product(v,c)
# print(f"{affiche}")


# Product using the numPy library
product =np.dot(v,c)
print(f"{product}")


# Vectorization with NumPy VS LOOP with the dot_product function
np.random.seed(1)
a = np.random.rand(10000000)
b = np.random.rand(10000000)

tic = time.time()
#c = np.dot(a,b)
toc = time.time()

#print(f"np.dot(a, b) =  {c:.4f}")
print(f"Vectorized version duration: {1000*(toc-tic):.4f} ms ")

tic = time.time()  # capture start time
#c = dot_product(a,b)
toc = time.time()  # capture end time

#print(f"my_dot(a, b) =  {c:.4f}")
print(f"loop version duration: {1000*(toc-tic):.4f} ms ")

del(a);del(b)  #remove these big arrays from memory


### Matrix creation 
a = np.zeros((1,5))
print(f"a shape = {a.shape}, a = {a}")

# matrice de 2 examples comprenant chacune 1 feature
a = np.zeros((2, 1))                                                                   
print(f"a shape = {a.shape}, a = {a}") 

# matrice de 3 examples comprenant chacune 3 features
a = np.random.random_sample((3, 3))  
print(f"a shape = {a.shape}, a = {a}") 

# NumPy routines which allocate memory and fill with user specified values
a = np.array([[5], [4], [3]]);   print(f" a shape = {a.shape}, np.array: a = {a}")
a = np.array([[5],   # One can also
              [4],   # separate values
              [3]]); #into separate rows
print(f" a shape = {a.shape}, np.array: a = {a}")

#print (f"{a[1]}")
print("SWITCHING")
#vector indexing operations on matrices
a = np.arange(6).reshape(-1, 2)   #reshape is a convenient way to create matrices
print(f"a.shape: {a.shape}, \n a=\n {a}")

#access an element
print(f"\na[2,0].shape:   {a[2, 0].shape}, a[2,0] = {a[2, 0]},     type(a[2,0]) = {type(a[2, 0])} Accessing an element returns a scalar\n")

#vector 2-D slicing operations
a = np.arange(20).reshape(-1, 5)
#a = a.reshape(2,10)
print(f"a = \n{a}")

#access 5 consecutive elements (start:stop:step)
print("a[0, 2:7:1] = ", a[0, 2:7:1], ",  a[0, 2:7:1].shape =", a[0, 2:7:1].shape, "a 1-D array")

#access 5 consecutive elements (start:stop:step) in two rows
print("a[:, 2:7:1] = \n", a[:, 2:7:1], ",  a[:, 2:7:1].shape =", a[:, 2:7:1].shape, "a 2-D array")

# access all elements
print("a[:,:] = \n", a[:,:], ",  a[:,:].shape =", a[:,:].shape)

# access all elements in one row (very common usage)
print("a[1,:] = ", a[1,:], ",  a[1,:].shape =", a[1,:].shape, "a 1-D array")
# same as
print("a[1]   = ", a[1],   ",  a[1].shape   =", a[1].shape, "a 1-D array")
