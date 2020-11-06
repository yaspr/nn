import YNN
import time
import array
import numpy as np

#Python array reduction
def reduction(a):

    r = 0.0
    
    for i in range(0, len(a)):
        r += a[i]

    return r

#Number of array elements (4 Bytes x n = size in Bytes)
n = 1000000

#
print("Array size (B):   ", (n * 4))
print("Array size (KiB): ", (n * 4.0) / (1024.0))
print("Array size (MiB): ", (n * 4.0) / pow(2, 20))
print("\n")

#
a = np.random.rand(n)
b = np.random.rand(n)

#
aa = array.array('f', a)
bb = array.array('f', b)

#
before = time.perf_counter()
print ("NUMPY v:\t\t", np.sum(aa))
after = time.perf_counter()
elapsed1 = (after - before)
print ("NUMPY t:\t\t", elapsed1, "\n")

#
before = time.perf_counter()
print ("PYTHON v:\t\t", reduction(aa))
after = time.perf_counter()
elapsed6 = (after - before)
print ("PYTHON t:\t\t", elapsed6, "\n")

#
before = time.perf_counter()
print ("REDUC v:\t\t", YNN.reduc_f32(aa))
after = time.perf_counter()
elapsed2 = (after - before)
print ("REDUC t:\t\t", elapsed2, "\n")

before = time.perf_counter()
print ("REDUCO v:\t\t", YNN.reduc_f32_optimized(aa))
after = time.perf_counter()
elapsed3 = (after - before)
print ("REDUCO t:\t\t", elapsed3, "\n")

print ("REDUC / REDUCO: ", elapsed2 / elapsed3, "\n")

#
before = time.perf_counter()
print ("DOTPROD v:\t\t", YNN.dotprod_f32(aa, bb))
after = time.perf_counter()
elapsed4 = (after - before)
print ("DOTPROD t:\t\t", elapsed4, "\n")

before = time.perf_counter()
print ("DOTPRODO v:\t\t", YNN.dotprod_f32_optimized(aa, bb))
after = time.perf_counter()
elapsed5 = (after - before)
print ("DOTPRODO t:\t\t", elapsed5, "\n")

print ("DOTPROD / DOTPRODO: ", elapsed4 / elapsed5, "\n")

print ("NUMPY / REDUC:\t\t", elapsed1 / elapsed2)
print ("NUMPY / REDUCO:\t\t", elapsed1 / elapsed3)

print ("PYTHON / REDUC:\t\t", elapsed6 / elapsed2)
print ("PYTHON / REDUCO:\t\t", elapsed6 / elapsed3)
