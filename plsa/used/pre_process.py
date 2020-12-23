import numpy as np

def zentai(array: np.ndarray) -> np.ndarray:
    return array/np.sum(array)

def kl_diver(A: np.ndarray, B: np.ndarray) -> float:
    pre = (A * np.ma.log(A)).sum()
    later = (A * np.ma.log(B)).sum()
    '''
    A_after = A.astype(np.float64)
    AB = np.divide(A_after, B, out=np.zeros_like(B), where=(B != 0))
    print(AB)
    aab = A*np.ma.log(AB)
    print(aab.sum())
    '''
    return pre - later
