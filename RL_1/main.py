import numpy as np
import sys
import matplotlib.pyplot as plt
np.set_printoptions(threshold=sys.maxsize)


def runEpisode(policy,gamma):
    
    
    d0 = {
        's1' : 0.6,
        's2' : 0.3,
        's3' : 0.1   
    }

    transition_probabilities = {
        's1': {
            'a1': {'s4': 1.0},
            'a2': {'s4': 1.0}
        },
        's2': {
            'a1': {'s4': 0.8, 's5': 0.2},
            'a2': {'s4': 0.6, 's5': 0.4}
        },
        's3': {
            'a1': {'s4': 0.9, 's5': 0.1},
            'a2': {'s5': 1.0}
        },
        's4': {
            'a1': {'s6': 1.0},
            'a2': {'s6': 0.3, 's7': 0.7}
        },
        's5': {
            'a1': {'s6': 0.3, 's7': 0.7},
            'a2': {'s7': 1.0}
        }
    }
    
    reward_function = {
        's1': {'a1': 7, 'a2': 10},
        's2': {'a1': -3, 'a2': 5},
        's3': {'a1': 4, 'a2': -6},
        's4': {'a1': 9, 'a2': -1},
        's5': {'a1': -8, 'a2': 2}
    }
       
    discounted_return = 0
    
    
    
    ### Sample S0 ###    
    current_state = np.random.choice(list(d0.keys()), p=list(d0.values()))    
    #print(f"Sampled current state: {current_state}")
    
    ### Agent-Environment Interaction
    for i in range(2):
    
        ## sample action from current_state
        current_state_policy = policy[current_state]
        action = np.random.choice(list(current_state_policy.keys()), p=list(current_state_policy.values()))
        #print(f"Current action: {action}")
        
        ## get reward and add to the discounted_return       
        reward = reward_function[current_state][action]
        #print("reward calculated after gamma multiplication = ",x)
        discounted_return += reward*(gamma**(i))
        #print(f"discounted_return =  {discounted_return}")
        
        ## get next state and set current_state to next state
        next_state_transitions = transition_probabilities[current_state][action]
        next_state = np.random.choice(list(next_state_transitions.keys()), p=list(next_state_transitions.values()))
        #print(f"next state =  {next_state}")
        current_state = next_state   
    
    ##################################
    

    return discounted_return

def policyGenerator():

    s = ['s1','s2','s3','s4','s5']
    actions = ['a1', 'a2']
    
    policy_generated = {
        's1': {'a1': 0, 'a2': 0},
        's2': {'a1': 0, 'a2': 0},
        's3': {'a1': 0, 'a2': 0},
        's4': {'a1': 0, 'a2': 0},
        's5': {'a1': 0, 'a2': 0}
    }
    
    for state in s:
        a = np.random.choice(actions)
        policy_generated[state][a] = 1
        
    return policy_generated                 

if __name__ == '__main__':

 
 
 

    policy = {
        's1': {'a1': 0.5, 'a2': 0.5},
        's2': {'a1': 0.7, 'a2': 0.3},
        's3': {'a1': 0.9, 'a2': 0.1},
        's4': {'a1': 0.4, 'a2': 0.6},
        's5': {'a1': 0.2, 'a2': 0.8}
    }
    
    gamma_test = 0.9
    
    #print(policy[2][1])
    
    
    ######################################################
    ####################### Question 2a ##################
    ######################################################
    
    estimates = []
    returns_stored = []
    sum_to_t = 0
    num_simulations = 150000
    for simulations in range(num_simulations):       
        discounted_return = runEpisode(policy,gamma_test)
        returns_stored.append(discounted_return)
        sum_to_t += discounted_return
        estimates.append(sum_to_t/(simulations+1))
        #print(f"discounted_return = {discounted_return}")
        
        





 
    ######################################################
    ####################### Question 2b ##################
    ######################################################
    
    print("\n####################### Question 2b #######################")
    print("The Average Discounted Return after executing 150000 simulations is: ",np.round(estimates[149999],4))
    #print("Avg calculated with numpy = ",np.mean(returns_stored))
    print("Variance of the return of all simulations = ",np.round(np.var(returns_stored),4))
    print("\n")
    
 
    ######################################################
    ####################### Question 2c ##################
    ###################################################### 
    
    print("####################### Question 2c #######################")
    gamma_2c =[0.25, 0.5, 0.75, 0.99]
    for g in gamma_2c:
        returns_stored_2c = []     
        for simulations in range(num_simulations):     
            discounted_return = runEpisode(policy,g)
            returns_stored_2c.append(discounted_return)
        print("Average Discounted Return for gamma '"+str(g)+"' is = ",np.round(np.mean(returns_stored_2c),4))
        
    print("\n")

  
    ######################################################
    ####################### Question 2d ##################
    ######################################################  
    
    performance_estimates_xplot = []
    best_policy = {}
    best_policy_performance = np.NINF
    for pol in range(250):
        #### Generate new deterministic policy ####
        policy_2d = policyGenerator()
        #print(policy_2d)
        #returns_stored_2d = 0
        sum_to_t_2d = 0
        estimates_2d = []
        for episode in range(100):
            discounted_return = runEpisode(policy_2d,gamma_test)
           # returns_stored_2d.append(discounted_return)
            sum_to_t_2d += discounted_return
            estimates_2d.append(sum_to_t_2d/(episode+1))
        perf  = np.mean(estimates_2d)
        if perf > best_policy_performance:
            best_policy_performance = perf 
            best_policy = policy_2d
            performance_estimates_xplot.append(perf)
        else:
            performance_estimates_xplot.append(performance_estimates_xplot[-1])
    
    #print("Best policy = ",best_policy)
    #print("Best perf = ", best_policy_performance)
    
    


    
    
    ######################################################
    ################### Plot figures #####################
    ######################################################
    
    ########### 2a ##################
    
    custom_x_ticks = np.arange(0, 150001, 15000)
    plt.figure(1,figsize=(10, 6))
    plt.plot(range(1, num_simulations + 1), estimates)
    plt.xlabel("Number of Episodes")
    plt.ylabel("Estimate J^(π)")
    plt.xticks(custom_x_ticks)
    #plt.yticks(custom_y_ticks)
    plt.title("Estimate of J^(π) Over Episodes")
    plt.grid(True)

    
    
    ########### 2d ###################
    
    custom_x_ticks = np.arange(0, 251, 10)
    plt.figure(2,figsize=(10, 6))
    plt.plot(range(1, 251), performance_estimates_xplot)
    plt.xlabel("Policy(N)")
    plt.ylabel("Performance Estimate J^(π) of best policy")
    plt.xticks(custom_x_ticks)
    #plt.yticks(custom_y_ticks)
    plt.title("Performance Estimate J^(π) of best policy over N = 250")
    plt.grid(True)
    
    
    
    plt.show()
