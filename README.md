# Visualize the training of an RBF network
Train an RBF network on different datasets (options: moons, circles, linear, and, or, xor). The number of hidden units determines how many prototype vectors are fitted. All parameters of the RBF network are trained with gradient descent instead of initializing prototype vectors randomly or with k-means centroids and optimizing the remaining parameters with gradient descent. If the learning rate is too high, training will become unstable.

## Usage
```
python rbfn.py [-h] [-d {moons,circles,linear,and,or,xor}] [-hu HIDDEN_UNITS] [-lr LEARNING_RATE] [-n N_SAMPLES]

optional arguments:
  -h, --help            show this help message and exit
  -d {moons,circles,linear,and,or,xor}, --dataset {moons,circles,linear,and,or,xor}
                        the dataset used for training
  -hu HIDDEN_UNITS, --hidden_units HIDDEN_UNITS
                        the number of hidden units (number of prototype
                        vectors)
  -lr LEARNING_RATE, --learning_rate LEARNING_RATE
                        the learning rate used in the gradient descent steps
  -n N_SAMPLES, --n_samples N_SAMPLES
                        the number of training samples

```
