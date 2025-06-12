import numpy as np 

arr = np.load("/ptmp/noka/tusz_new/analysis/alpha_z50_labels.npy")
arr_sliced = arr[:1000]

print(arr_sliced)