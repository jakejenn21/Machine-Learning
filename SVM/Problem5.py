import numpy as np
import pandas as pd

#global params

C = 1/3
N = 3.0

#step 1

w1= [0.0,0.0,0.0,0.0] 
w1 = np.array(w1)
w2= [0.0,0.0,0.0,0.0]
w2 = np.array(w2)

lr = 0.01

xi = [1.0,0.5,-1.0,0.3]
yi = 1.0


coeff = (lr * C * N * yi)

f = w1-lr*w2
s = np.multiply(xi,coeff)

w_final1 = f + s

print ("\n\nsubgradient step one: ", w_final1)
print ("\n")

#step 2

w1= w_final1 
w_final1[0]=0.0
w2 = w_final1

lr = 0.005

xi = [1.0,-1.0,-2.0,-2.0]
yi = -1.0


coeff = (lr * C * N * yi)

f = w1-lr*w2
s = np.multiply(xi,coeff)

w_final2 = f + s


print ("subgradient step two: ", w_final2)
print ("\n")

#step 2

w1= w_final2
w_final2[0]=0.0
w2 = w_final2

lr = 0.0025

xi = [1.0,1.5,0.2,-2.5]
yi = 1.0


coeff = (lr * C * N * yi)

f = w1-lr*w2
s = np.multiply(xi,coeff)

w_final3 = f + s

print ("subgradient step three: ", w_final3)
print ("\n")

