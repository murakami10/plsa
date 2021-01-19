import numpy as np

def zentai(array: np.ndarray) -> np.ndarray:
    return array/np.sum(array)

def kl_diver(A: np.ndarray, B: np.ndarray) -> float:

    ab = np.divide(A, B, out=np.zeros_like(A), where=(B != 0))
    ab = np.ma.log(ab)
    ab = ab.sum()

    '''
    a = np.ma.log(A)
    b = np.ma.log(B)
    ab = a - b
    ab = ab.sum()
    '''

    '''
    pre = (np.ma.log(A)).sum()
    later = (np.ma.log(B)).sum()
    '''

    '''
    A_after = A.astype(np.float64)
    AB = np.divide(A_after, B, out=np.zeros_like(B), where=(B != 0))
    print(AB)
    aab = A*np.ma.log(AB)
    print(aab.sum())
    '''
    return ab

def process_result(array: np.ndarray) -> np.ndarray:
    return array.sum(axis=0)/array.shape[0]
