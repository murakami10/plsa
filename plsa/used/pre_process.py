import numpy as np

def zentai(array: np.ndarray) -> np.ndarray:
    return array/np.sum(array)

def kl_diver(A: np.ndarray, B: np.ndarray) -> float:

    ab = np.divide(A, B, out=np.zeros_like(A), where=(B != 0))
    ab = np.ma.log(ab)
    ab = ab.sum()

    return ab

def euclid(A: np.ndarray, B: np.ndarray) -> float:
    ab = A - B
    ab = ab*ab
    ab = np.sum(ab)

    return ab

def euclid_ver2(A: np.ndarray, B: np.ndarray) -> float:
    ab = A - B
    ab = ab * ab
    ab = ab / A
    ab = np.sum(ab)

    return ab



def process_result(array: np.ndarray) -> np.ndarray:
    return array.sum(axis=0)/array.shape[0]
