import numpy as np
import pickle 



def pos_calc(angle = 0, distance = 0):

    
    
    angle = angle * np.pi/180
    x = distance*np.sin(angle) # opposite side x.
    y = distance*np.cos(angle) # adjasent side y. 6/arcsin = x
   
    
    y = np.sqrt(y**2)
    y = y*(-1)
    
    #return angle, distance #used for force calculation in force_calculator.py
    return x,y # used along with ur_data by next_state to calculate a trajectory.

    
if __name__ == '__main__':
    x,y = pos_calc()
