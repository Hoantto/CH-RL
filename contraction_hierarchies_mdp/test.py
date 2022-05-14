"""
Reinforcement learning for contraction hierarchies algorithm.
Author:     Hongzheng Bai
Date:       2022/05/03

"""
from cProfile import run
from contraction_hierarchies import CH
from RL_brain import DeepQNetwork

def run_ch_mdp():
    step = 0
    for episode in range(100):
        # initial observation
        observation =  env.reset()
        
        while True:
            # RL choose action based on observation
            action = RL.choose_action(observation)
            
            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)
            
            RL.store_transition(observation, action, reward, observation_)
            
            if (step > 200) and (step % 30 == 0):
                RL.learn()
                
            # swap observation
            observation = observation_
            
            # break while loop when end of this episode
            if done:
                print("done round:",episode+1)
                env.ch_rest()
                break
            step += 1
    
    
    # end of ch-RL
    print('ch-RL over')
            
            
            
        

if __name__ == "__main__":
    env = CH("/home/bhz/source_code/road-network/road-minnesota.txt",weight=False)
    RL = DeepQNetwork(env.n_action, env.n_features,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=200,
                      memory_size=int(len(env.G.nodes)/2),
                        # memory_size=4000,
                    #   output_graph=True
                      )
    run_ch_mdp()
    env.query_test(10)
    # env.performance_contrast()
    env.performance_contrast_mdp()
    RL.plot_cost()
    