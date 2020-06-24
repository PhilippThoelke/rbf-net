import numpy as np
import tensorflow as tf
from sklearn import datasets
from matplotlib import pyplot as plt
import argparse

parser = argparse.ArgumentParser(description='''Train an RBF network on different datasets.
The number of hidden units determines how many prototype vectors are fitted.
All parameters of the RBF network are trained with gradient descent
instead of initializing prototype vectors randomly or with k-means centroids
and optimizing the remaining parameters with gradient descent.
If the learning rate is too high, training will become unstable.''')

parser.add_argument('-d', '--dataset', type=str, default='moons', required=False,
					choices=['moons', 'circles', 'linear', 'and', 'or', 'xor'],
					help='the dataset used for training')
parser.add_argument('-hu', '--hidden_units', type=int, default=5, required=False,
					help='the number of hidden units (number of prototype vectors)')
parser.add_argument('-lr', '--learning_rate', type=float, default=2, required=False,
					help='the learning rate used in the gradient descent steps')
parser.add_argument('-n', '--n_samples', type=int, default=100, required=False,
					help='the number of training samples')
args = parser.parse_args()

def make_dataset(kind):
	'''Generate a dataset'''
	x_train = np.random.uniform(-1, 1, size=(args.n_samples, 2))
	x_int = np.round(x_train / 2 + 0.5).astype(np.bool)
	if kind == 'and':
		y_train = x_int[:,0] & x_int[:,1]
	elif kind == 'or':
		y_train = x_int[:,0] | x_int[:,1]
	elif kind == 'xor':
		y_train = x_int[:,0] ^ x_int[:,1]
	else:
		raise ValueError(f'Unsupported dataset type "{kind}"')
	return x_train, y_train.astype(np.float)

# training data
if args.dataset == 'moons':
	x_train, y_train = datasets.make_moons(args.n_samples, noise=0.1)
elif args.dataset == 'circles':
	x_train, y_train = datasets.make_circles(args.n_samples, noise=0.1, factor=0.6)
elif args.dataset == 'linear':
	x_train = np.random.uniform(-1, 1, size=(args.n_samples, 2))
	y_train = (x_train[:,0] >= 0).astype(np.float)
else:
	x_train, y_train = make_dataset(args.dataset)

data_xlims = x_train[:,0].min(), x_train[:,0].max()
data_ylims = x_train[:,1].min(), x_train[:,1].max()

# hyperparameters
lr = args.learning_rate

n_in = 2
n_hidden = args.hidden_units
n_out = 1

plot_margin = 0.15
plot_resolution = 100

# initialize RBF network variables
xi = tf.Variable(np.random.normal(size=(n_in, n_hidden)))
sigmas = tf.Variable(np.ones(n_hidden))
w_out = tf.Variable(np.random.normal(size=(n_hidden, n_out)))
b_out = tf.Variable(np.zeros(n_out))

def predict(x):
	'''Compute output of the RBF network using the previously defined variables'''
	dist = tf.linalg.norm(np.stack([x] * xi.shape[1], axis=-1) - xi, axis=1)
	return tf.nn.sigmoid(tf.exp(-0.5 * sigmas * dist) @ w_out + b_out)

def acc(xs, ys):
	'''Compute the accuracy of the RBF network'''
	return np.mean(ys == np.round(predict(xs)[:,0]))

def get_extent():
	'''Compute the limits of prototype vectors and data plus a certain margin'''
	xmin = min(xi.numpy()[0].min(), data_xlims[0]) - plot_margin
	xmax = max(xi.numpy()[0].max(), data_xlims[1]) + plot_margin
	ymin = min(xi.numpy()[1].min(), data_ylims[0]) - plot_margin
	ymax = max(xi.numpy()[1].max(), data_ylims[1]) + plot_margin
	return xmin, xmax, ymin, ymax

def activation_map(extent):
	'''Compute the activation map for a given range of inputs'''
	xvals = np.linspace(*extent[:2], plot_resolution)
	yvals = np.linspace(*extent[:-3:-1], plot_resolution)
	grid = np.meshgrid(xvals, yvals)
	samples = np.stack(grid, axis=-1).reshape(-1, 2)
	return grid, predict(samples).numpy().reshape(plot_resolution, plot_resolution)

# get limits
extent = get_extent()
grid, act = activation_map(extent)

# create the plot
plt.ion()
fig, ax = plt.subplots()
tit = ax.set_title(f'Accuracy: {acc(x_train, y_train):.2%}')
im = ax.imshow(act, extent=extent, cmap='coolwarm', vmin=0, vmax=1)
scat1 = ax.scatter(*x_train[y_train == 0].T)
scat2 = ax.scatter(*x_train[y_train == 1].T)
prot_vec_scat = ax.scatter(*xi.numpy(), marker='x', s=50)
cont = ax.contour(*grid, act, levels=[0.5], extent=extent, colors=['black'])
fig.canvas.draw()
fig.canvas.flush_events()

# training loop
variables = [xi, sigmas, w_out, b_out]
while plt.fignum_exists(fig.number):
	with tf.GradientTape() as tape:
		# get the loss for the current epoch
		loss = tf.reduce_mean((y_train - predict(x_train)[:,0]) ** 2, axis=0)
		# compute gradients
		grad = tape.gradient(loss, variables)
		# apply gradients
		for var, g in zip(variables, grad):
			var.assign_sub(g * lr)

	# compute new limits
	extent = get_extent()
	grid, act = activation_map(extent)

	# update visualization
	tit.set_text(f'Accuracy: {acc(x_train, y_train):.2%}')
	im.set_extent(extent)
	im.set_data(act)
	[c.remove() for c in cont.collections]
	cont = ax.contour(*grid, act, levels=[0.5], extent=extent, colors=['black'])
	scat1.set_offsets(x_train[y_train == 0])
	scat2.set_offsets(x_train[y_train == 1])
	prot_vec_scat.set_offsets(xi.numpy().T)
	ax.set(xlim=extent[:2], ylim=extent[2:])
	fig.canvas.draw()
	fig.canvas.flush_events()
