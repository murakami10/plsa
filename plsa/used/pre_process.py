import numpy as np
from typing import List

def zentai(array: np.ndarray) -> np.ndarray:
    return array/np.sum(array)

def kl_diver(A: np.ndarray, B: np.ndarray) -> float:

    ab = np.divide(A, B, out=np.zeros_like(A), where=(B != 0))
    ab = np.ma.log(ab)
    ab = ab.sum()

    return ab

def kl_diver_with_label(A: np.ndarray, B: np.ndarray, lable: List[int]) -> float:
    C: np.ndarray = np.zeros_like(B)
    for index, l in enumerate(lable):
        C[:, l] += B[:, index]
    ab = np.divide(A, C, out=np.zeros_like(A), where=(C != 0))
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
