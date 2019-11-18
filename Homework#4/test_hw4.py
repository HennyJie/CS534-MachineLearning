'''
@Author: your name
@Date: 2019-11-17 21:38:42
@LastEditTime: 2019-11-17 21:45:25
@LastEditors: Please set LastEditors
@Description: In User Settings Edit
@FilePath: /Machine Learning/Homework#4/test_hw4.py
'''
import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(12, 3.6))
# compare AUC with/without standardization in changing the number of features t (k=5)
plt.subplot(131)
plt.plot(range(30), range(0, 300, 10),
         color="r", linestyle="-", marker="*", linewidth=1, label="without standardized")
plt.plot(range(30), range(5, 305, 10),
         color="g", linestyle="-", marker="+", linewidth=1, label="with standardized")
plt.title("AUROC for top t features \n (k=5)")
plt.xlabel("t")
plt.ylabel("AUROC")
plt.legend(loc='best')

# compare AUC with/without standardization in changing the number of neighbors k (t=30)
plt.subplot(132)
plt.plot(range(30), range(0, 300, 10),
         color="r", linestyle="-", marker="*", linewidth=1, label="without standardized")
plt.plot(range(30), range(5, 305, 10),
         color="g", linestyle="-", marker="+", linewidth=1, label="with standardized")
plt.title("AUROC for top k neighbors \n (all features)")
plt.xlabel("k")
plt.ylabel("AUROC")
plt.legend(loc='best')

# # compare AUC with/without standardization in changing the number of neighbors k
# (t= the number of features that reached highest AUROC)
plt.subplot(133)
plt.plot(range(30), range(0, 300, 10),
         color="r", linestyle="-", marker="*", linewidth=1, label="without standardized")
plt.plot(range(30), range(5, 305, 10),
         color="g", linestyle="-", marker="+", linewidth=1, label="with standardized")
plt.title("AUROC for top k neighbors \n (features achieved max auroc)")
plt.xlabel("k")
plt.ylabel("AUROC")
plt.legend(loc='best')

plt.tight_layout()
plt.show()
