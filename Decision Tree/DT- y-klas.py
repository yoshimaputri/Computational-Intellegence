import random
import pprint
from csv import reader
import math
import numpy as np

def load_csv (filename):
	data = list (reader (open (filename, 'rt')))
	for row in data:
		for column in range (len (row)):
			row[column] = float (row[column].strip ())
	return data, len (data[0])

def normalize(data):
	for i in range(len(data[1]) - 1):
		arr = []
		for x in data:
			arr.append(x[i])
		for x in data:
			x[i] = (x[i] - np.min(arr)) / (np.max(arr) - np.min(arr)) 
	return data

def cross_validation_split (dataset, nfold):
	splits = []
	copy = list (dataset)
	fold_size = len (dataset) // nfold # for not float
	for i in range (nfold):
		fold = []
		while len (fold) < fold_size:
			index = random.randrange (len (copy))
			fold.append (copy.pop (index))
		splits.append (fold)
	return splits

def accuracy_metric (actual, predicted):
	correct = 0
	TP = 0 # true positive
	FP = 0 # false positive
	FN = 0 # false negative
	TN = 0 # true negative
	for i in range (len (actual)):
		# print(actual[i], predicted[i])
		if actual[i] == 1 and predicted[i] == 1:
			TP += 1 # benar dikatakan positif
		if actual[i] == 0 and predicted[i] == 1:
			FP += 1 # salah dikatakan positif
		if actual[i] == 1 and predicted[i] == 0:
			FN += 1 # salah dikatakan negatif
		if actual[i] == 0 and predicted[i] == 0:
			TN += 1 # benar dikatakan positif
		if actual[i] == predicted[i]:
			correct += 1
	print('\n---------------------------')
	print('     Pos     Neg')
	print('T [TP:', TP,' FN:', FN, ']')
	print('F [FP:', FP,' TN:', TN, ']')
	acc = correct / float(len(actual)) * 100.0
	if (TP + FP) != 0:
		prec = float(TP / (TP + FP)) * 100.0
	else:
		prec = 0
	if (TP + FN) != 0:
		recall = float(TP / (TP + FN)) * 100.0
	else:
		recall = 0
	return acc, prec, recall
	
def evaluate_algorithm (data_set, algorithm, nfold, *args):
	# data_set = normalize(data_set)
	folds = cross_validation_split (data_set, nfold)
	accuracy = []
	precision = []
	recall = []
	i = 0
	for fold in folds:
		train_set = list (folds)
		train_set.remove (fold)
		train_set = sum (train_set, [])
		test_set = []
		for row in fold:
			row_copy = list (row)
			test_set.append (row_copy)
			row_copy[-1] = None
		i += 1
		predicted = algorithm (train_set, test_set, *args)
		actual = [row[-1] for row in fold]
		acc, precs, rec = accuracy_metric(actual, predicted)
		print()
		print('Result Fold[', i, ']')
		print('- Accuracy : ', acc, '%')
		print('- Precision : ', precs, '%')
		print('- Recall : ', rec, '%')
		accuracy.append(acc)
		precision.append(precs)
		recall.append(rec)
	return accuracy, precision, recall

def test_split(index, value, dataset):
	left, right = list(), list()
	for row in dataset:
		if row[index] < value:
			left.append(row)
		else:
			right.append(row)
	return left, right

def gini_index (score):
	return 1.0 - score

def entropy_index (score):
	return math.sqrt (score) * -1.0 * math.log2 (math.sqrt (score))

def misclassification_error (score):
	pass

def index_function (groups, classes, method):
	# total attr per kelas
	attr = float (sum ([len (group) for group in groups]))
	index = 0.0
	for group in groups:
		size = float (len (group))
		if size == 0:
			continue
		score = 0.0
		max_prop = 0
		# hitung total kelas 0 dan 1 (index)
		for class_val in classes:
			p = [row[-1] for row in group].count (class_val) / size
			max_prop = max (max_prop, p)
			score += (p * p)
		# sesuai method
		if method == misclassification_error:
			index = 1 - (max_prop / attr)
		else: #gini & entropy
			index += (method (score)) * (size / attr)
	return index

# mencari gini terbaik/ entropy terbaik/ me terbaik
def get_split (parent, dataset, method):
	class_values = list(set(row[-1] for row in dataset))
	b_index, b_value, b_score, b_groups = 999, 999, 999, None
	for index in range(len(dataset[0])-1):
		for row in dataset:
			groups = test_split(index, row[index], dataset) 
			if method == gini_index or parent is None:
				score = index_function (groups, class_values, method)
			else:
				# print((parent['value']))
				score = parent['value'] - index_function (groups, class_values, method)
			if abs (score) < (b_score):
				b_index, b_value, b_score, b_groups = index, row[index], score, groups
	return {'index':b_index, 'value':b_value, 'groups':b_groups}

# create leaf node
def leaf (group):
	outcomes = [row[-1] for row in group]
	return max (set (outcomes), key=outcomes.count)

# child node sbg node baru atau leaf node
def split (node, max_depth, min_size, method, depth=1):
	left, right = node['groups']
	del(node['groups'])
	# check for a no split
	if not left or not right:
		node['left'] = node['right'] = leaf(left + right)
		return
	# check for max depth
	if depth >= max_depth:
		node['left'], node['right'] = leaf (left), leaf (right)
		return
	# process left child
	if len (left) <= min_size:
		node['left'] = leaf (left)
	else:
		node['left'] = get_split (node, left, method)
		split (node['left'], max_depth, min_size, method, depth + 1)
	# process right child
	if len(right) <= min_size:
		node['right'] = leaf(right)
	else:
		node['right'] = get_split (node, right, method)
		split(node['right'], max_depth, min_size, method, depth + 1)

# Build a decision tree
def build_tree(train, max_depth, min_size, method):
	root = get_split (None, train, method)
	split (root, max_depth, min_size, method)
	# print (pprint.pformat (root, indent = 4, width = 100))
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

def decision_tree(train, test, max_depth, min_size, method):
	tree = build_tree(train, max_depth, min_size, method)
	predictions = list()
	for row in test:
		prediction = predict(tree, row)
		predictions.append(prediction)
	return(predictions)

def main ():
	data, attr_count = load_csv ('pima-indians-diabetes.data')
	fold = int(input('Enter the fold : '))
	print()
	print ('Impurity\n0. Gini Index\n1. Entropy\n2. Misclassification Error\n')
	method = [gini_index, entropy_index, misclassification_error][int (input ('Choose the number of Impurity Node : '))]

	max_depth = int(input('Enter the maximum depth of tree : '))
	min_size = int(input('Enter the minimum amount data in a node : '))
	
	acc, precision, recall = evaluate_algorithm (data, decision_tree, fold, max_depth, min_size, method)
	print()
	print ('Mean Accuracy : ', (sum (acc)/float (len (acc))), '%')
	print ('Mean Precision : ', (sum (precision) / float (len (precision))), '%')
	print ('Mean Recall : ', (sum (recall) / float (len (recall))), '%')

main()