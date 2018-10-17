import os

# opt_type_list = ['Adam', 'Adadelta', 'Adagrad', 'Ftrl', 'SGD', 'Momentum', 'Nesterov', 'RMSprop']
opt_type_list = ['Ftrl']

for opt_type in opt_type_list:
    os.system('python optimizer_test.py '+opt_type)

