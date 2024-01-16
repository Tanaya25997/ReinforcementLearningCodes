import numpy as np
import sys
import matplotlib.pyplot as plt
np.set_printoptions(threshold=sys.maxsize)



def Value_Iteration(gamma,reward,goals):

    ### State array
    state_values = [[0 for j in range(5)] for i in range(5)]

 
    #### Policy
    policy = [[None] * 5 for _ in range(5)]

    #print(policy)
    
    theta = 0.0001
    

    
    itr = 0
    
    while True:
        itr += 1
        delta = 0
        state_values_temp =  [[0 for j in range(5)] for i in range(5)]
        for r in range(0,5):
            for c in range(0,5):
                current_state_value = state_values[r][c]
                UP = 0
                DOWN = 0
                LEFT = 0
                RIGHT = 0
                if goals == 2:
                    if (r == 2 and c == 2) or (r == 3 and c == 2) or (r == 4 and c == 4) or (r == 0 and c == 2):
                        continue 
                elif (r == 2 and c == 2) or (r == 3 and c == 2) or (r == 4 and c == 4):
                    continue
                
                ####################################
                ######## CALCULATE ACTION UP #######
                ####################################
                
                # with 0.1 remain in the same state
                UP = 0.1*(reward[r][c] + gamma*state_values[r][c])
                
                # with 0.8 go to the intended state
                if r-1 > -1:
                    if (r-1==3 and c==2):
                        UP += 0.8*(reward[r][c] + gamma*state_values[r][c])
                    else:
                        UP += 0.8*(reward[r-1][c] + gamma*state_values[r-1][c])
                else:
                    UP +=  0.8*(reward[r][c] + gamma*state_values[r][c])  #### We hit the wall and remain in the same state
                    
                # with 0.05 go to right
                if c+1 < 5:
                    if (r==2 and c+1==2) or (r==3 and c+1==2):
                        UP += 0.05*(reward[r][c] + gamma*state_values[r][c])
                    else:
                        UP += 0.05*(reward[r][c+1] + gamma*state_values[r][c+1]) 
                else:
                    UP += 0.05*(reward[r][c] + gamma*state_values[r][c])   #### We hit the wall and remain in the same state

                # with 0.05 go to left
                if c-1 > -1:
                    if (r==2 and c-1==2) or (r==3 and c-1==2):
                        UP += 0.05*(reward[r][c] + gamma*state_values[r][c])
                    else:               
                        UP += 0.05*(reward[r][c-1] + gamma*state_values[r][c-1])
                else:
                    UP += 0.05*(reward[r][c] + gamma*state_values[r][c])  #### We hit the wall and remain in the same state
                
        
                
                ####################################
                ##### CALCULATE ACTION DOWN ########
                ####################################
                
                # with 0.1 remain in the same state
                DOWN = 0.1*(reward[r][c] + gamma*state_values[r][c])
                
                # with 0.8 go to the intended state
                if r+1 < 5:
                    if (r+1==2 and c==2):
                        DOWN += 0.8*(reward[r][c] + gamma*state_values[r][c])
                    else:
                        DOWN += 0.8*(reward[r+1][c] + gamma*state_values[r+1][c])
                else:
                    DOWN += 0.8*(reward[r][c] + gamma*state_values[r][c])        #### We hit the wall and remain in the same state
                    
                # with 0.05 go to right
                if c+1 < 5:
                    if (r==2 and c+1==2) or (r==3 and c+1==2):
                        DOWN += 0.05*(reward[r][c] + gamma*state_values[r][c])
                    else:
                        DOWN += 0.05*(reward[r][c+1] + gamma*state_values[r][c+1])
                else:
                    DOWN += 0.05*(reward[r][c] + gamma*state_values[r][c])       #### We hit the wall and remain in the same state
                
                # with 0.05 go tp left
                if c-1 > -1:
                    if (r==2 and c-1==2) or (r==3 and c-1==2):
                        DOWN += 0.05*(reward[r][c] + gamma*state_values[r][c])
                    else:
                        DOWN += 0.05*(reward[r][c-1] + gamma*state_values[r][c-1])
                else:
                    DOWN += 0.05*(reward[r][c] + gamma*state_values[r][c])       #### We hit the wall and remain in the same state
                
        
                ####################################
                ##### CALCULATE ACTION LEFT ########
                ####################################

                # with 0.1 remain in the same state
                LEFT = 0.1*(reward[r][c] + gamma*state_values[r][c])
                
                # with 0.8 go to the intended state
                if c-1 > -1:
                    if (r==2 and c-1==2) or (r==3 and c-1==2):
                        LEFT += 0.8*(reward[r][c] + gamma*state_values[r][c])
                    else:
                        LEFT += 0.8*(reward[r][c-1] + gamma*state_values[r][c-1])
                else:
                    LEFT += 0.8*(reward[r][c] + gamma*state_values[r][c])
                    
                # with 0.05 go up
                if r-1 > -1:
                    if (r-1==3 and c==2):
                        LEFT += 0.05*(reward[r][c] + gamma*state_values[r][c])
                    else:
                        LEFT += 0.05*(reward[r-1][c] + gamma*state_values[r-1][c])
                else:
                    LEFT += 0.05*(reward[r][c] + gamma*state_values[r][c])
                    
                # with 0.05 go down
                if r+1 < 5:
                    if (r+1==2 and c==2):
                        LEFT += 0.05*(reward[r][c] + gamma*state_values[r][c])
                    else:
                        LEFT += 0.05*(reward[r+1][c] + gamma*state_values[r+1][c]) 
                else:
                    LEFT += 0.05*(reward[r][c] + gamma*state_values[r][c])
                        
                
           
                ####################################
                ##### CALCULATE ACTION RIGHT #######
                ####################################
 
                # with 0.1 remain in the same state
                RIGHT = 0.1*(reward[r][c] + gamma*state_values[r][c])
                
                # with 0.8 go to the intended state
                if c+1 < 5:
                    if (r==2 and c+1==2) or (r==3 and c+1==2):
                        RIGHT += 0.8*(reward[r][c] + gamma*state_values[r][c])
                    else:
                        RIGHT += 0.8*(reward[r][c+1] + gamma*state_values[r][c+1])
                else:
                    RIGHT += 0.8*(reward[r][c] + gamma*state_values[r][c])
                    
                # with 0.05 go up
                if r-1 > -1:
                    if (r-1==3 and c==2):
                        RIGHT += 0.05*(reward[r][c] + gamma*state_values[r][c])
                    else:
                        RIGHT += 0.05*(reward[r-1][c] + gamma*state_values[r-1][c])
                else:
                    RIGHT += 0.05*(reward[r][c] + gamma*state_values[r][c])

                # with 0.05 go down
                if r+1 < 5:
                    if (r+1==2 and c==2):
                        RIGHT += 0.05*(reward[r][c] + gamma*state_values[r][c])
                    else:
                        RIGHT += 0.05*(reward[r+1][c] + gamma*state_values[r+1][c]) 
                else:
                    RIGHT += 0.05*(reward[r][c] + gamma*state_values[r][c])

               
                ###################################################
                #### Assign max as the next value of the state 
                ###################################################
                
                state_values_temp[r][c], policy[r][c] = max((UP, 'UP'), (DOWN, 'DOWN'), (LEFT, 'LEFT'), (RIGHT, 'RIGHT'))
                state_values_temp[r][c] = round(state_values_temp[r][c],4)            
                delta = max(delta,abs(current_state_value-state_values_temp[r][c]))
        
        state_values = state_values_temp
        #print("\n state_values = \n",state_values)
        if delta < theta:
            break
            
            
            
    return state_values,policy,itr
    
if __name__ == '__main__':


    arrow_symbols = {
    'UP': '↑',
    'DOWN': '↓',
    'LEFT': '←',
    'RIGHT': '→'
    }
    
    ##########################################################
    ####################   QUESTION 2.1 ######################
    ##########################################################
    
    reward = [[0 for j in range(5)] for i in range(5)]
    reward[4][2] = -10
    reward[4][4] = 10
    
    print("\n######################################################")
    print("##################### QUESTION 2.1 ###################")
    print("######################################################")
    
    gamma = 0.9
    state_values,policy,itr =  Value_Iteration(gamma,reward,1)      
        
  
    print("Iterations taken to converge = ",itr)
    
    print("\nValue Function")
    for row in state_values:
        print('  '.join(['{:.4f}'.format(element) for element in row]))
        

    
    print("\nPolicy")
    
    #print(policy)
    
    for i, row in enumerate(policy):
        policy_symbols = [arrow_symbols[action] if action is not None else ' ' for action in row]
        if i == len(policy) - 1:
            policy_symbols[-1] = 'G'
        
        print('  '.join(policy_symbols))
    
    ##########################################################
    ####################   QUESTION 2.2 ######################
    ##########################################################
    
    
    print("\n######################################################")
    print("##################### QUESTION 2.2 ###################")
    print("######################################################")
    
    gamma_2 = 0.25
    state_values_2,policy_2,itr_2 =  Value_Iteration(gamma_2,reward,1)      
        
  
    print("Iterations taken to converge = ",itr_2)
    
    print("\nValue Function")
    for row in state_values_2:
        print('  '.join(['{:.4f}'.format(element) for element in row]))
        
    
    print("\nPolicy")
    
    #print(policy)
    
    for i, row in enumerate(policy_2):
        policy_symbols_2 = [arrow_symbols[action] if action is not None else ' ' for action in row]
        if i == len(policy_2) - 1:
            policy_symbols_2[-1] = 'G'
        
        print('  '.join(policy_symbols_2))   

    ##########################################################
    ####################   QUESTION 2.3 ######################
    ##########################################################
    
    reward_3 = [[0 for j in range(5)] for i in range(5)]
    reward_3[4][2] = -10
    reward_3[4][4] = 10
    reward_3[0][2] = 5
    
    print("\n######################################################")
    print("##################### QUESTION 2.3 ###################")
    print("######################################################")
    
    gamma_3 = 0.9
    state_values_3,policy_3,itr_3 =  Value_Iteration(gamma_3,reward_3,1)      
        
  
    print("Iterations taken to converge = ",itr_3)
    
    print("\nValue Function")
    for row in state_values_3:
        print('  '.join(['{:.4f}'.format(element) for element in row]))
        
    
    print("\nPolicy")
    
    #print(policy)
    
    for i, row in enumerate(policy_3):
        policy_symbols_3 = [arrow_symbols[action] if action is not None else ' ' for action in row]
        if i == len(policy_3) - 1:
            policy_symbols_3[-1] = 'G'
        
        print('  '.join(policy_symbols_3))  
        
        
    ##########################################################
    ####################   QUESTION 2.4 ######################
    ##########################################################
    
    
    print("\n######################################################")
    print("##################### QUESTION 2.4 ###################")
    print("######################################################")
    
    gamma_4 = 0.9
    state_values_4,policy_4,itr_4 =  Value_Iteration(gamma_4,reward_3,2)      
        
  
    print("Iterations taken to converge = ",itr_4)
    
    print("\nValue Function")
    for row in state_values_4:
        print('  '.join(['{:.4f}'.format(element) for element in row]))
        
    
    print("\nPolicy")
    
    #print(policy)
    
    for i, row in enumerate(policy_4):
        policy_symbols_4 = [arrow_symbols[action] if action is not None else ' ' for action in row]
        if i == 0:
            policy_symbols_4[2] = 'G'
        if i == len(policy_4) - 1:
            policy_symbols_4[-1] = 'G'
        
        print('  '.join(policy_symbols_4)) 
        
        
    print("\nAfter experimenting, checking with gamma = 0.91320501 which avoids s(0,2) altogether \n")
    
    gamma_5 = 0.91320501
    state_values_5,policy_5,itr_5 =  Value_Iteration(gamma_5,reward_3,2)      
        
  
    print("Iterations taken to converge = ",itr_5)
    
    print("\nValue Function")
    for row in state_values_5:
        print('  '.join(['{:.4f}'.format(element) for element in row]))
        
    
    print("\nPolicy")
    
    #print(policy)
    
    for i, row in enumerate(policy_5):
        policy_symbols_5 = [arrow_symbols[action] if action is not None else ' ' for action in row]
        if i == 0:
            policy_symbols_5[2] = 'G'
        if i == len(policy_5) - 1:
            policy_symbols_5[-1] = 'G'
        
        print('  '.join(policy_symbols_5)) 
        
    
       
    ##########################################################
    ####################   QUESTION 2.5 ######################
    ##########################################################
    
    reward_4 = [[0 for j in range(5)] for i in range(5)]
    reward_4[4][2] = -10
    reward_4[4][4] = 10
    reward_4[0][2] = 4.48445
    
    print("\n######################################################")
    print("##################### QUESTION 2.5 ###################")
    print("######################################################")
    
    gamma_6 = 0.9
    state_values_6,policy_6,itr_6 =  Value_Iteration(gamma_6,reward_4,2)      
        
  
    print("Iterations taken to converge = ",itr_6)
    
    print("\nValue Function")
    for row in state_values_6:
        print('  '.join(['{:.4f}'.format(element) for element in row]))
        
    
    print("\nPolicy")
    
    #print(policy)
    
    for i, row in enumerate(policy_6):
        policy_symbols_6 = [arrow_symbols[action] if action is not None else ' ' for action in row]
        if i == 0:
            policy_symbols_6[2] = 'G'
        if i == len(policy_6) - 1:
            policy_symbols_6[-1] = 'G'
        
        print('  '.join(policy_symbols_6)) 
        
    
