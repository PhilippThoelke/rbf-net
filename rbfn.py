import numpy as np
import tensorflow as tf
from scipy import spatial
from sklearn import datasets, cluster
from matplotlib import pyplot as plt
import argparse

parser = argparse.ArgumentParser(description='''Train an RBF network on different datasets.
The number of hidden units determines how many prototype vectors are fitted.
You can choose how the prototype vectors should be initialized. The options are
backpropagation (default), k-means centroids or random training samples.
If the learning rate is too high, training might become unstable.''')

parser.add_argument('-d', '--dataset', type=str, default='moons', required=False,
					choices=['moons', 'circles', 'linear', 'and', 'or', 'xor'],
					help='the dataset used for training')
parser.add_argument('-nh', '--hidden_units', type=int, default=5, required=False,
					help='the number of hidden units (number of prototype vectors)')
parser.add_argument('-lr', '--learning_rate', type=float, default=2, required=False,
					help='the learning rate used in the gradient descent steps')
parser.add_argument('-n', '--n_samples', type=int, default=100, required=False,
					help='the number of training samples')
parser.add_argument('-po', '--prototype_optimization', type=str, default='backpropagation',
					required=False, choices=['backpropagation', 'kmeans', 'random'],
					help='the optimization/initialization technique for the prototype vectors')
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

def max_dist(vectors):
	'''Compute the maximum distance between the vectors given in vectors (n_vecs, n_dims)'''
	return spatial.distance_matrix(vectors, vectors).max()

# initialize prototype vectors
if args.prototype_optimization == 'backpropagation':
	xi = tf.Variable(np.random.normal(size=(n_in, n_hidden)))
	sigmas = tf.Variable(np.ones(n_hidden))
elif args.prototype_optimization == 'kmeans':
	xi = cluster.KMeans(n_clusters=n_hidden).fit(x_train).cluster_centers_.T
	# set sigma to the maximum distnace between the prototype vectors
	sigmas = np.ones(n_hidden) * max_dist(xi.T)
elif args.prototype_optimization == 'random':
	xi = np.random.permutation(x_train)[:n_hidden].T
	# set sigma to the maximum distnace between the prototype vectors
	sigmas = np.ones(n_hidden) * max_dist(xi.T)
else:
	raise ValueError(f'Unsupported optimization method "{args.prototype_optimization}"')

# initialize RBF network variables
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
	xmin = min(np.min(xi[0]), data_xlims[0]) - plot_margin
	xmax = max(np.max(xi[0]), data_xlims[1]) + plot_margin
	ymin = min(np.min(xi[1]), data_ylims[0]) - plot_margin
	ymax = max(np.max(xi[1]), data_ylims[1]) + plot_margin
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
xi_numpy = xi if type(xi) == np.ndarray else xi.numpy()
prot_vec_scat = ax.scatter(*xi_numpy, marker='x', s=50)
cont = ax.contour(*grid, act, levels=[0.5], extent=extent, colors=['black'])
fig.canvas.draw()
fig.canvas.flush_events()

# create a list of trainable variables
variables = [w_out, b_out]
if args.prototype_optimization == 'backpropagation':
	variables += [xi, sigmas]

# training loop
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
	xi_numpy = xi if type(xi) == np.ndarray else xi.numpy()
	prot_vec_scat.set_offsets(xi_numpy.T)
	ax.set(xlim=extent[:2], ylim=extent[2:])
	fig.canvas.draw()
	fig.canvas.flush_events()
