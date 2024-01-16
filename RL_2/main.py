import numpy as np
import sys
import matplotlib.pyplot as plt
np.set_printoptions(threshold=sys.maxsize)



def Evolution_Strategies(n,nPerturbations,sigma,alpha,N,M,trials,representation_type):


  
    
    
    
    
    # Create the n x n identity matrix     
    I = np.eye(n)
    
    # Initial policy parameter vector, θ0 in R^n
    theta = np.random.rand(n)
    #print(theta.shape)
    
    
    Average_performance = []
    
    ### for t trials ####
    for t in range(0,trials):
        J_avg_over_trial = []
        J_epsilon_i = np.zeros(n)
        ##### Run n perturbations of a policy####
        for i in range(0,nPerturbations):
            epsilon_i = np.random.multivariate_normal(mean=np.zeros(n), cov=I)
            #print(epsilon_i.shape)
            policy = theta + sigma*epsilon_i
            ### estimate return ###
            J = estimate_J(policy,N,M,representation_type)
            J_avg_over_trial.append(J)
            #print("J= ",J)
            J_epsilon_i += J*epsilon_i 
        
        #print("Average Performance after trial ", str(t+1)," = ",np.mean(J_avg_over_trial))
        Average_performance.append(np.mean(J_avg_over_trial))
        #### get new improved policy parameters
        theta = theta + (alpha*J_epsilon_i)/(sigma*nPerturbations)

     
    return theta, Average_performance
    
def estimate_J(policy,N,M,representation_type):



    
    G = []
    for episodes in range(0,N):
        G.append(Evaluate_policy(policy,representation_type,M))
        
    return np.mean(G)




def Evaluate_policy(policy,representation_type,M):



    gamma = 1
    reward = 1

    #### Initial state (x,v,w,w.)
    x = 0
    v = 0
    w = 0
    w_dot = 0
    
         
    g = 9.8         #(gravity)
    mc = 1.0        #(cart’s mass)
    mp = 0.1        #(pole’s mass)
    mt = mc + mp    #(total mass)
    l = 0.5         #(pole’s length)
    tau = 0.02        #(agent executes an action every 0.02 seconds)


 
    R = 0

    for timesteps in range(0,500):
    
        terms = np.arange(1, M + 1)
        
        
        ##### Normalize state #######
        if representation_type == "cosine":
            # Normalize to [0, 1]
            normalized_x = (x + 2.4) / 4.8
            normalized_w = (w + (np.pi / 15)) / (2 * (np.pi / 15))
            # Normalize v and ω̇ using their typical intervals. Initially capping to [0,1] since interval is not known
            normalized_v = (v + 4) / 8  # Clip the value to [0, 1]
            normalized_w_dot = (w_dot + 5) / 10  # Clip the value to [0, 1]
            
            '''
            normalized_v = max(0, min(1, v))  # Clip the value to [0, 1] intially
            normalized_w_dot = max(0, min(1, w_dot))  # Clip the value to [0, 1]
            '''
            
                # Create the state feature vector using cosine terms
            feature_vector = np.concatenate([
            np.ones(1),  # The first term is always 1
            np.cos(terms * np.pi * normalized_x),
            np.cos(terms * np.pi * normalized_v),
            np.cos(terms * np.pi * normalized_w),
            np.cos(terms * np.pi * normalized_w_dot)
            ])
            
            feature_vector.reshape(-1, 1)
        
        elif representation_type == "sine":
            # Normalize to [-1, 1]
            normalized_x = (2*(x + 2.4)/4.8) - 1
            normalized_w = (2*(w + (np.pi / 15)))/(2 * (np.pi / 15)) - 1
            normalized_v =(2*(v + 4)/8) - 1 # Clip the value to [0, 1]
            normalized_w_dot =  (2*(w_dot + 5)/10) - 1 # Clip the value to [0, 1]
            '''
            normalized_v = max(-1, min(1, (v+1)/2))  # Clip the value to [0, 1]
            normalized_w_dot = max(-1, min(1, (w_dot+1)/2))  # Clip the value to [0, 1]
            '''
            
            
            feature_vector = np.concatenate([
            np.ones(1),  # The first term is always 1
            np.sin(terms * np.pi * normalized_x),
            np.sin(terms * np.pi * normalized_v),
            np.sin(terms * np.pi * normalized_w),
            np.sin(terms * np.pi * normalized_w_dot)
            ]) 
            
            feature_vector.reshape(-1, 1)
            
            
          
        threshold = np.dot(feature_vector,policy)
        
        if threshold <= 0:
            F = -10
        else:
            F = 10 
            
            
            
            
            
        #### Calculate reward #####
        
        R += gamma**(timesteps) * reward
        ##### Recalculate values for next time step #####
        
        b = (F + mp*l*np.sin(w)*w_dot**(2))/mt 
        c = (g*np.sin(w) - np.cos(w)*b)/l*((4/3) - (mp*(np.cos(w))**(2))/mt)
        d = b - (mp*l*c*np.cos(w))/mt
    

        x = x + tau*v 
        v = v + tau*d 
        #print("velocity = ", v)
        w = w + tau*w_dot
        w_dot = w_dot + tau*c 
        #print("w_dot = ", w_dot)
        
        
        if x<-2.4 or x>2.4 or w<-np.pi/15 or w>np.pi/15:
            #print("limit crossed. timesteps = ", timesteps)
            return R
        
           
    return R








if __name__ == '__main__':

 
    
    #### Setting hyperparameters ######  
    n = [41, 41, 41, 25, 37]                     # n = 4M + 1, Consider M = 3
    nPerturbations = [50, 50, 50, 100, 75]       #the number of perturbations you perform on the policy to create nPerturbations policies 
    sigma = [0.5, 0.5, 1, 0.75, 0.4]           # σ =  Exploration parameter
    alpha = [0.01, 0.1, 0.01, 0.02, 0.01]          # α = Step size
    N = [10, 10, 20, 10, 8]                       # Number of episodes 
    M = [10, 10, 10, 6, 9]                         #### order of the Fourier basis. Note: the higher the value of M, the more accurately is the signal/system represented
     
    
    ######################################################
    ################# QUESTION 1 #########################
    ######################################################
    
    print("\n####################### Question 2.1 #######################")
    #print("\n####################### Running cosine 5 trials #######################")

    trials_1 = 5
    
    '''
    #### representation_type = cosine #####
    representation_type = 'cosine' 
    Average_performance_COSINE = []
    for i in range(0,5):
        theta_COSINE,Average_performance_cos = Evolution_Strategies(n[i],nPerturbations[i],sigma[i],alpha[i],N[i],M[i],trials_1, representation_type)
        print(Average_performance_cos)
        Average_performance_COSINE.append(Average_performance_cos)
    print(Average_performance_COSINE)
    
    '''
    
    print("\nRunning sine 5 trials ") 
    
    #### representation_type = sine #####
    representation_type = 'sine' 
    Average_performance_SINE = []
    for i in range(0,5):
        theta_SINE,Average_performance_sin= Evolution_Strategies(n[i],nPerturbations[i],sigma[i],alpha[i],N[i],M[i],trials_1, representation_type)
        #print(Average_performance_sin)
        Average_performance_SINE.append(Average_performance_sin)
    
    '''
    marker = 'o'
    line_style = '-'
    plt.figure(1,figsize=(10, 6))
    for p_cos in range(0,5):
        std_dev_cos_1 = np.std(Average_performance_COSINE[p_cos],axis=0)
        #plt.errorbar(range(1, trials_1 + 1), Average_performance_COSINE[p_cos], yerr=std_dev_cos_1, fmt='-o')
        plt.plot(range(1, trials_1 + 1), Average_performance_COSINE[p_cos], marker, linestyle = line_style)
        plt.fill_between(range(1, trials_1 + 1), Average_performance_COSINE[p_cos] - std_dev_cos_1, Average_performance_COSINE[p_cos] + std_dev_cos_1, alpha=0.2, color='lightblue', label="Std Dev")
    plt.xlabel("Number of Iteration/Updates performed by ES")
    plt.ylabel("Average Return/Performance")
    plt.xticks(np.arange(0, trials_1 + 1, 1))
    plt.title("Learning Curve Plot for 5 trials/runs for Cosine")
    plt.legend()
    #plt.grid(True)
    '''
    
    marker = 'o'
    line_style = '-'
    #colour = ['lightred','orange','green','red','purple']
    plt.figure(2,figsize=(10, 6))
    for p_sin in range(0,5):
        #std_dev_sin_1 = np.std(Average_performance_SINE[p_sin],axis=0)
        #plt.errorbar(range(1, trials_1 + 1), Average_performance_SINE[p_sin], yerr=std_dev_sin_1, fmt='-o')
        plt.plot(range(1, trials_1 + 1), Average_performance_SINE[p_sin], marker, linestyle = line_style, label= f'Hyperparameter {p_sin+1}')
        #plt.fill_between(range(1, trials_1 + 1), Average_performance_SINE[p_sin] - std_dev_sin_1, Average_performance_SINE[p_sin] + std_dev_sin_1, alpha=0.2, color='lightblue', label="Std Dev")
    plt.xlabel("Number of Iteration/Updates performed by ES")
    plt.ylabel("Average Return/Performance")
    plt.xticks(np.arange(0, trials_1 + 1, 1))
    plt.title("Learning Curve Plot for 5 trials/runs for sine")
    plt.legend(loc='upper left')
    #plt.grid(True)
    
       
    
    ######################################################
    ################# QUESTION 2 #########################
    ######################################################
    
    print("\n####################### Question 2.2 #######################")
    
    trials_2 = 20
    
    
    '''
    print("\n####################### Running cosine 20 trials #######################")
    
    #### representation_type = cosine #####
    representation_type = 'cosine' 
    Average_performance_COSINE_2 = []
    for i in range(0,5):
        theta_COSINE_2,Average_performance_cos_2 = Evolution_Strategies(n[i],nPerturbations[i],sigma[i],alpha[i],N[i],M[i],trials_2, representation_type)
        print(Average_performance_cos_2)
        Average_performance_COSINE_2.append(Average_performance_cos_2)
    #print(Average_performance_COSINE)
    '''
    
    
    
    print("\nRunning sine 20 trials") 
    #### representation_type = sine #####
    representation_type = 'sine' 
    Average_performance_SINE_2 = []
    for i in range(0,5):
        theta_SINE_2,Average_performance_sin_2= Evolution_Strategies(n[i],nPerturbations[i],sigma[i],alpha[i],N[i],M[i],trials_2, representation_type)
        #print(Average_performance_sin_2)
        Average_performance_SINE_2.append(Average_performance_sin_2)
    
    

    '''
    marker = 'o'
    line_style = '-'
    plt.figure(3,figsize=(10, 6))
    for p_cos_2 in range(0,5):
        std_dev_cos = np.std(Average_performance_COSINE_2[p_cos_2],axis=0)
        plt.errorbar(range(1, trials_2 + 1), Average_performance_COSINE_2[p_cos_2], yerr=std_dev_cos, fmt='-o')
        #plt.plot(range(1, trials_2 + 1), Average_performance_COSINE_2[p_cos_2], marker, linestyle = line_style, label='Hyperparam [n=1]')
    plt.xlabel("Number of Iteration/Updates performed by ES")
    plt.ylabel("Average Return/Performance")
    plt.xticks(np.arange(0, trials_2 + 1, 1))
    plt.title("Learning Curve Plot for 20 trials/runs for Cosine")
    plt.legend()
    #plt.grid(True)
    '''
    
    
    marker = 'o'
    line_style = '-'
    plt.figure(4,figsize=(10, 6))
    for p_sin_2 in range(0,5):
        std_dev_sin_2 = np.std(Average_performance_SINE_2[p_sin_2],axis=0)
        #plt.errorbar(range(1, trials_2 + 1), Average_performance_COSINE_2[p_cos_2], yerr=std_dev_sin, fmt='-o')
        plt.plot(range(1, trials_2 + 1), Average_performance_SINE_2[p_sin_2], marker, linestyle = line_style,  label= f'Hyperparameter {p_sin_2+1}')
        plt.fill_between(range(1, trials_2 + 1), Average_performance_SINE_2[p_sin_2] - std_dev_sin_2, Average_performance_SINE_2[p_sin_2] + std_dev_sin_2, alpha=0.2, color='lightblue', label="Std Dev")
    plt.xlabel("Number of Iteration/Updates performed by ES")
    plt.ylabel("Average Return/Performance")
    plt.xticks(np.arange(0, trials_2 + 1, 1))
    plt.title("Learning Curve Plot for 20 trials/runs for sine")
    plt.legend(loc='upper left')
    #plt.grid(True)
 
    
    
    plt.show()
