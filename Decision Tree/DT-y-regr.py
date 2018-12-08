import csv
import math
import numpy as np
from sklearn.metrics import mean_squared_error
from random import seed
from random import randrange
from sklearn.metrics import average_precision_score

def normalize(dataset, params):
	X = np.array(dataset)
	a_min = []
	a_max = []
	for i in range(params):
		a_min.append(X[:, i ].min())
		a_max.append(X[:, i ].max())
	for index in range(len(dataset)):
		for i in range(params-1):
			dataset[index][i] = (dataset[index][i] - a_min[i])/(a_max[i]-a_min[i])

def load_csv(filename):
	with open(filename, 'rt', encoding='utf-8') as csvfile:
		lines = csv.reader(csvfile)
		files = list(lines)
		dataset = []
		for row in files :
			for element in row :
				parts = element.split()
				for x in range(len(parts)):
					parts[x] = float(parts[x])
				dataset.append(parts)
	return dataset

def cross_validation_split(dataset, n_folds):
	dataset_split = list()
	dataset_copy = list(dataset)
	fold_size = int(len(dataset) // n_folds) # for not float
	for i in range(n_folds):
		fold = list()
		while len(fold) < fold_size:
			index = randrange(len(dataset_copy))
			fold.append(dataset_copy.pop(index))
		dataset_split.append(fold)
	return dataset_split

def accuracy_metric(actual, predicted):
	return math.sqrt(mean_squared_error(actual, predicted))

def evaluate_algorithm(dataset, n_folds , treshold):
	folds = cross_validation_split(dataset, n_folds)
	RMSE = []
	# precision = []
	# recall = []
	i = 0
	for fold in folds:
		train_set = list(folds)
		train_set.remove(fold)
		train_set = sum(train_set, [])
		test_set = list()
		for row in fold:
			row_copy = list(row)
			test_set.append(row_copy)
			row_copy[-1] = None
		i += 1
		predicted = decision_tree(train_set, test_set, treshold)
		actual = [row[-1] for row in fold]
		# scores.append(accuracy_metric(actual, predicted))
		rmse = accuracy_metric(actual, predicted)
		print('Result Fold[', i, ']')
		print('- RMSE : ', rmse, '%')
		# print('- Precision : ', precs, '%')
		# print('- Recall : ', rec, '%')
		RMSE.append(rmse)
		# precision.append(precs)
		# recall.append(rec)
	return RMSE

# Split a dataset based on an attribute and an attribute value
def test_split(index, value, dataset):
	left, right = list(), list()
	for row in dataset:
		if row[index] < value:
			left.append(row)
		else:
			right.append(row)
	return left, right

# Calculate the standard deviation reduction for a split dataset
def calculate_std_dev(groups, classes):
	total = float(sum([len(group) for group in groups]))
	sdr = np.std(np.array(classes))
	stddev = 0.0
	for group in groups:
		size = float(len(group))
		if size == 0:
			continue
		# score the group based on the score for each class
		stddev += np.std(np.array([row[-1] for row in group])) * size / total
	return sdr - stddev

# Select the best split point for a dataset
def get_split_sdr(dataset):
	class_values = list(row[-1] for row in dataset)
	b_index, b_value, b_score, b_groups = -999, -999, -999, None
	for index in range(len(dataset[0])-1):
		for row in dataset:
			groups = test_split(index, row[index], dataset)
			stddev = calculate_std_dev(groups, class_values)
			if stddev > b_score:
				b_index, b_value, b_score, b_groups = index, row[index], stddev, groups
	return {'index':b_index, 'value':b_value, 'groups':b_groups}

# node treshold in dataset
def tresholding(dataset):
	class_values = np.array(list(row[-1] for row in dataset))
	stddev = np.std(class_values)
	mean = np.mean(class_values)
	return stddev / mean

# Create a terminal node value
def leaf(group):
	outcomes = [row[-1] for row in group]
	return np.mean(np.array(outcomes))

# Create child splits for a node or make terminal
def split(node, treshold):
	left, right = node['groups']
	del(node['groups'])
	# check for a no split
	if not left or not right:
		node['left'] = node['right'] = leaf(left + right)
		return
	# process left child
	if tresholding(left) <= treshold:
		node['left'] = leaf(left)
	else:
		node['left'] = get_split_sdr(left)
		split(node['left'],treshold)
	# process right child
	if tresholding(right) <= treshold:
		node['right'] = leaf(right)
	else:
		node['right'] = get_split_sdr(right)
		split(node['right'],treshold)

# Build a decision tree
def create_root(train, treshold):
	root = get_split_sdr(train)
	split(root, treshold)
	return root

# Make a prediction with a decision tree
def predict(node, row):
	if row[node['index']] < node['value']:
		if isinstance(node['left'], dict):
			return predict(node['left'], row)
		else:
			return node['left']
	else:
		if isinstance(node['right'], dict):
			return predict(node['right'], row)
		else:
			return node['right']

# Regression Tree Algorithm
def decision_tree(train, test, treshold):
	tree = create_root(train, treshold)
	# build_tree(tree, 0, 'ROOT')
	print()
	predictions = list()
	for row in test:
		prediction = predict(tree, row)
		predictions.append(prediction)
	return(predictions)

def build_tree(tree, depth, param):
	classes = ["CRIM","ZN","INDUS","CHAS","NOX","RM","AGE","DIS","RAD","TAX","PTRATIO","B","LSTAT","MEDV"]
	if type(tree) == np.float64 :
		print('-'*(depth+1) + repr(param) + ' ' + 'Leaf = ' + repr(tree))
		return
	print('-'*depth + repr(param) + ' ' +  repr(classes[tree['index']]) + repr(tree['value']))
	build_tree(tree['left'], depth+1, 'Left')    
	build_tree(tree['right'], depth+1, 'Right')
	
def main ():
	data = load_csv ('housings.data')
	fold = int(input('Enter the fold : '))
	treshold = int(input('Enter the node treshold (%) : '))
	treshold /= 100
	rmse = evaluate_algorithm (data, fold, treshold)
	print()
	print ('Mean RMSE : ', (sum (rmse)/float (len (rmse))), '%')
	# print ('Mean Precision : ', (sum (precision) / float (len (precision))), '%')
	# print ('Mean Recall : ', (sum (recall) / float (len (recall))), '%')

main()