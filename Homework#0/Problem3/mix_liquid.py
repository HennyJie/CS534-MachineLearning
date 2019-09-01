import numpy as np
import pandas as pd 

# matrix representation of one cycle
choco_to_orange = np.array([[0.7, 0], [0.3, 1]])
orange_to_choco = np.array([[1, 0.2], [0, 0.8]])
one_cycle = np.dot(orange_to_choco, choco_to_orange)

state = {}
state[0] = np.array([1, 1])
data_frame = pd.DataFrame(columns=['number_of_cycle', 'volume_of_chocolate_cup', 'volume_of_orange_cup'])

# simulate the liquid mixing process
for i in range(1, 101):
    state[i] = np.dot(one_cycle, state[i-1])
    if i==1 or i==10:
        print("After Cycle {}, Chocolate cup has {} volume liquid".format(i, state[i][0])) 
        print("After Cycle {}, Orange cup has {} volume liquid".format(i, state[i][1])) 
    data_frame.loc[i-1] = [i, state[i][0], state[i][1]]

print(data_frame.to_string(index=False))

    