This repository contains the code for experiment on various loss functions and convergence of different optimization algorithms.

The code is simple and easy to understand. 

Experiment on various loss functions: Just open the *loss_function_experiment.ipynb* with the jupyter notebook and you can understand all the processes and logic.

Experiment on convergence of different optimization algorithms: You can use tensorbord to open the log file in logdir to see the convergence curve. The code is in *test_optimizer.py* and *optimizer_test.py*.


## Visualization

cross entropy:

![Fig1](./image/softmax.jpg)

center loss:

![Fig2](https://github.com/RDShi/loss_function/blob/master/image/center.jpg)

triplet loss [1,2]:

![Fig3](https://github.com/RDShi/loss_function/blob/master/image/triplet.jpg)


----
[1]Schroff F, Kalenichenko D, Philbin J. FaceNet: A unified embedding for face recognition and clustering[J]. 2015:815-823.

[2]Li C, Ma X, Jiang B, et al. Deep Speaker: an End-to-End Neural Speaker Embedding System[J]. 2017.
