import numpy as np
'''
ls = [[1,2,3,4],[5,6,7,8]]
nls = np.array(ls)
nls = nls/np.sum(nls, axis=1)[:,None]

print(nls)
'''

'''
a = np.arange(12).reshape(2,3,2)
b = np.arange(48).reshape(3,2,8)
c = a*b

print(c)
'''
'''
a = np.array([1, 2, 3, 4])
print(np.log(a))
'''

a = np.array([-1, 0, 1, 2, 3], dtype=float)
b = np.array([ 0, 0, 0, 2, 2], dtype=float)
c = np.divide(a, b, out=np.zeros_like(a), where=b!=0)
print(c)