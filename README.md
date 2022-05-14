# CH-RL
***
 Using **Reinforcement Learning methology** guides the construction of **Contraction Hierarchies Algorithm**.
***
## Recommendation of experiment environment
|environment|released version|
|---|---|
|networkx|2.5.1|
|python|3.6.2|
|numpy|1.19.5|
|pandas|0.23.4|
|tensorflow|1.13.0|
***
## Tutorial
* For better performance boost, implementation requires **GPU support**
* In order to deploy the experimental environment more conveniently, it is recommended to use **conda** to quickly install packages in the **Linux** environment.
* ***Steps***
    > (1) Complete the experimental environment configuration of the local or remote server.

    > (2) Modify the road network path name in the file <u>.\contraction_hierarchies_mdp\test.py</u> and use the absolute path to indicate the path of the road network dataset to be trained.

    > (3) Run <u>.\contraction_hierarchies_mdp\test.py</u> and analyze tje query performance improvements using reinforcement learning.

    > (4) You can test the performance of the recursive contraction hierarchies algorithm using only the **ipynb** file in folder <u>.\contraction_hierarchies</u>.
    
