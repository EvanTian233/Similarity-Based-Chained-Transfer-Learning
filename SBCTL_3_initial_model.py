# Authors:	Yifang Tian , Ljubisa Sehovac
# Email:	ytian285@uwo.ca, lsehovac@uwo.ca

# Step 3 in Similarity-Based Chained Transfer Learning algorithm
# Using the forecasting algorithm to build initial model with the forecasting set of the initial meter
# The structures and weights are saved for transfer learning
# Need to take care about all the dir while using, look for "os.chdir" in the code

# PyTorch Machine Learning framework was used to build the model and compute results
# PyTorch can be installed from the terminal,visit https://pytorch.org/ and choose the correct configuation.


# importing the needed libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import init

import random
import time

import math



###########################################################################################################################

# building the S2S model

class S2S_Model(nn.Module):
	def __init__(self, cell_type, input_size, hidden_size, use_cuda):
		super(S2S_Model, self).__init__()

		self.input_size = input_size
		self.hidden_size = hidden_size

		self.cell_type = cell_type

		# raise error if one of these three cells not chosen. No other cells exist yet.
		if self.cell_type not in ['rnn', 'gru', 'lstm']:
			raise ValueError(self.cell_type, " is not an appropriate cell type. Please select one of rnn, gru, or lstm.")

		if self.cell_type == 'rnn':
			self.Ecell = nn.RNNCell(self.input_size, self.hidden_size)
			self.Dcell = nn.RNNCell(1, self.hidden_size)

		if self.cell_type == 'gru':
			self.Ecell = nn.GRUCell(self.input_size, self.hidden_size)
			self.Dcell = nn.GRUCell(1, self.hidden_size)

		if self.cell_type == 'lstm':
			self.Ecell = nn.LSTMCell(self.input_size, self.hidden_size)
			self.Dcell = nn.LSTMCell(1, self.hidden_size)


		self.lin_usage = nn.Linear(self.hidden_size, 1)

		self.use_cuda = use_cuda

		self.init()


	# VERY IMPORTANT INIT PARAMS FUNCTIONS***
	# function to intialize weight parameters. If you would like to analyze the initial weight parameters, uncomment out the j's.
	# More info on weight parameters can be found at pytorch nn documentation:  https://pytorch.org/docs/stable/nn.html
	# Seach for GRUCell, RNNCell, or LSTMCell.
	def init(self):

		if self.cell_type == 'rnn' or self.cell_type == 'gru':
			#j = []
			for p in self.parameters():
				if p.dim() > 1:
					init.orthogonal_(p.data, gain=1.0)
					#j.append(p.data)
				if p.dim() == 1:
					init.constant_(p.data, 0.0)
					#j.append(p.data)

		elif self.cell_type == 'lstm':
			#j = []
			for p in self.parameters():
				if p.dim() > 1:
					init.orthogonal_(p.data, gain=1.0)
					#j.append(p.data)
				if p.dim() == 1:
					init.constant_(p.data, 0.0)
					init.constant_(p.data[self.hidden_size:2*self.hidden_size], 1.0)
					#j.append(p.data)
		#return j


	# This is the function that "consumes" the input data and learns from it
	def consume(self, x):

		# for rnn and gru cells
		if self.cell_type == 'rnn' or self.cell_type == 'gru':

			h = torch.zeros(x.shape[0], self.hidden_size)

			# you will see this if loop throughout the code. If cuda is True, it means use the GPU. If cuda is False, it means use the CPU
			if self.use_cuda:
				h = h.cuda()

			for T in range(x.shape[1]):
				h = self.Ecell(x[:, T, :], h)

			pred_usage = self.lin_usage(h)

		# for lstm cells
		elif self.cell_type == 'lstm':

			h0 = torch.zeros(x.shape[0], self.hidden_size)
			c0 = torch.zeros(x.shape[0], self.hidden_size)

			if self.use_cuda:
				h0 = h0.cuda()
				c0 = c0.cuda()

			h = (h0, c0)

			for T in range(x.shape[1]):
				h = self.Ecell(x[:, T, :], h)

			pred_usage = self.lin_usage(h[0])


		return pred_usage, h


	# This is the function that predicts the next N steps ahead
	def predict(self, pred_usage, h, target_length):
		# decoder forward function
		preds = []

		# for rnn and gru
		if self.cell_type == 'rnn' or self.cell_type == 'gru':

			for step in range(target_length):
				h = self.Dcell(pred_usage, h)
				pred_usage = self.lin_usage(h)

				preds.append(pred_usage.unsqueeze(1))

			preds = torch.cat(preds, 1)

		# for lstm
		elif self.cell_type == 'lstm':

			for step in range(target_length):
				h = self.Dcell(pred_usage, h)
				pred_usage = self.lin_usage(h[0])

				preds.append(pred_usage.unsqueeze(1))

			preds = torch.cat(preds, 1)


		return preds










#################################################################################################################################################

# main function

# INPUTS:
#	seed:		Can be any number. Random number initializer. This is used to randomize all numbers, samples, etc. but to remember the
#				order of randomization to be able to reproduce results.
#       filename:       This is the dataset to use. Make sure the code and file are in the same directory. If not, you can change the directory
#                       to where the file is.. Easiest just to have code and dataset in the same directory
#	cuda:		2 options, either True or False. If True use GPU, if False use CPU
#	cell_type:	3 options, either 'rnn', 'gru', or 'lstm' -- this just chooses which cell to use in the model
#	window_source_size:	Can be any reasonable number. Correlates to number of timesteps to use in consume function, before prediction.
#						This is T, from paper.
#	window_target_size: Can be any reasonable number. Correlates to number of timesteps to use in prediction function. This is N, from paper.
#						This is how many timesteps you would like to predict ahead.
#	epochs:		Can be any reasonable number. This tells the model how many iterations of training you would like. 10 epochs
#				was found to be a sufficient number of epochs. More can be used, but risk overfitting.
#	batch_size:	Can be any reasonable number. This tells the model how many random samples to use at once. The higher this number is, the
#				faster training is, but the worse accuracy is (very slightly). The smaller it is, the slower training is, but better accuracy
#				(again very slightly). Recommend using a power of 2 as the batch size -- hence 32, 64, 128, or 256.
# 	hs:			Can be any reasonable number. This tells the model how many hidden units to use in the respective cells. Try using powers of 2
#				here as well -- hence 32, 64, 128. The higher the hidden size, the slower the training, but usually slightly better accuracy
#       to_plot:        Can be either True or False. True will output plots, False will not
#       create_file:   Can be either True or False. True will create anomalie file, False will not
#       just_anoms_file: Can be either True or False. If you would like the csv file with only the anomalies, keep True, otherwise keep False

def main(seed, filename, cuda, cell_type, window_source_size, window_target_size, epochs, batch_size, hs, to_plot,
	 threshold_1, threshold_2, threshold_3, threshold_4, create_file, just_anoms_file):

	# start time to see how long the script takes
	t0 = time.time()

	# seed == given seed
	np.random.seed(seed)
	torch.manual_seed(seed)

	# do not need P_MTRID (0), P_MTRCHN (1), TIMESTAMP (2), or YEAR (3). NOTE: This is for dataset that has already been manipulated!
	print("Loading dataset...")
	d = np.loadtxt(filename, delimiter=",", skiprows=1, dtype=str) # this is where you would input your dataset. Make sure
	model_1 = os.path.splitext(filename)[0]

	print("Removing columns...")					# you are working in the correct directory
	dataset = d[:, 2:].astype(np.float32)

	dataset = pd.DataFrame(dataset)
	dataset.columns = ['month', 'day_of_year', 'day_of_month', 'weekday', 'weekend', 'holiday', 'hour', 'minute',
					   'season', 'usage']

	# Dropped the minute
	dataset = dataset.drop('minute',1)

	usage_actual = dataset['usage']

	mu_usage = dataset['usage'].mean()
	std_usage = dataset['usage'].std()

	dataset = dataset.values


	# Standardization. This is needed to be able to train the model -- 0 mean and unit var
	print("Transforming data to 0 mean and unit var")
	MU = dataset.mean(0) # 0 means take the mean of the column
	dataset = dataset - MU

	STD = dataset.std(0) # same with std here
	dataset = dataset / STD

	# 5 minutes between rows, from this respective dataset
	# use 1 hour (12 rows) to predict next half hour (6 rows), if T = 12 and N = 6
	print("Generating training and test data...")
	WINDOW_SOURCE_SIZE = window_source_size
	WINDOW_TARGET_SIZE = window_target_size


	# getting actual usage vector, aligning with predicted values vector. Aka remove first window_source_size and remaining
	usage_actual = usage_actual.values
	usage_actual = usage_actual[int(dataset.shape[0]*0.80):]
	usage_actual = usage_actual[WINDOW_SOURCE_SIZE:]

	# 80% of the data will be used to train the model
	# 20% is used to test the model
	train_source = dataset[:int(dataset.shape[0]*0.80)]
	test_source = dataset[int(dataset.shape[0]*0.80):]


	# This function generates the sample windows. Refer to paper for in-depth explanation
	def generate_windows(data):
		x_train = []
		y_usage_train = []

		x_test = []
		y_usage_test = []

		# for training data
		idxs = np.random.choice(train_source.shape[0]-(WINDOW_SOURCE_SIZE+WINDOW_TARGET_SIZE), train_source.shape[0]-(WINDOW_SOURCE_SIZE+WINDOW_TARGET_SIZE), replace=False)

		for idx in idxs:
			x_train.append(train_source[idx:idx+WINDOW_SOURCE_SIZE].reshape((1, WINDOW_SOURCE_SIZE, train_source.shape[1])) )
			y_usage_train.append(train_source[idx+WINDOW_SOURCE_SIZE:idx+WINDOW_SOURCE_SIZE+WINDOW_TARGET_SIZE, -1].reshape((1, WINDOW_TARGET_SIZE, 1)) )

		x_train = np.concatenate(x_train, axis=0) # make them arrays and not lists
		y_usage_train = np.concatenate(y_usage_train, axis=0)

		# for testing data
		idxs = np.arange(0, len(test_source)-(WINDOW_SOURCE_SIZE+WINDOW_TARGET_SIZE), WINDOW_TARGET_SIZE)

		for idx in idxs:
			x_test.append(test_source[idx:idx+WINDOW_SOURCE_SIZE].reshape((1, WINDOW_SOURCE_SIZE, test_source.shape[1])) )
			y_usage_test.append(test_source[idx+WINDOW_SOURCE_SIZE:idx+WINDOW_SOURCE_SIZE+WINDOW_TARGET_SIZE, -1].reshape((1, WINDOW_TARGET_SIZE, 1)) )

		x_test = np.concatenate(x_test, axis=0) # make them arrays and not lists
		y_usage_test = np.concatenate(y_usage_test, axis=0)

		return x_train, y_usage_train, x_test, y_usage_test


	X_train, Y_train_usage, X_test, Y_test_usage = generate_windows(dataset)
	print("Created {} train samples and {} test samples".format(X_train.shape[0], X_test.shape[0]))
	# should see a lot less test samples -- this is OK. Sliding window used for test set of N steps, rest assured, the whole test set is still tested on.

	idxs = np.arange(0, len(test_source)-(WINDOW_SOURCE_SIZE+WINDOW_TARGET_SIZE), WINDOW_TARGET_SIZE)
	remainder = len(test_source) - (idxs[-1] + WINDOW_SOURCE_SIZE+WINDOW_TARGET_SIZE)

	usage_actual = usage_actual[:-remainder]




	#################################################################################################################################################

	# call the model

	print("Creating model...")
	INPUT_SIZE = X_train.shape[-1]
	HIDDEN_SIZE = hs
	CELL_TYPE = cell_type

	model = S2S_Model(CELL_TYPE, INPUT_SIZE, HIDDEN_SIZE, use_cuda=cuda)

	if cuda:
		torch.cuda.manual_seed(seed)
		torch.cuda.manual_seed_all(seed)
		model.cuda()


	print("MODEL ARCHITECTURE IS: ")
	print(model)

	print("\nModel parameters are on cuda: {}".format(next(model.parameters()).is_cuda))

	# hyperparameters -- can be changed, but only change if confident
	opt = optim.Adam(model.parameters(), lr=1e-3)
	loss_fn = nn.MSELoss(reduction='sum')
	EPOCHES = epochs
	BATCH_SIZE = batch_size


	print("\nStarting training...")

	train_loss = []
	test_loss = []

	# for each epoch, model goes through entire dataset once.
	for epoch in range(EPOCHES):

		t_one_epoch = time.time()

		print("Epoch {}".format(epoch+1))

		total_usage_loss = 0

		##########################################################################################################################################

		# TRAINING

		# for each batch, model learns and weights are updated
		for b_idx in range(0, X_train.shape[0], BATCH_SIZE):

			x = torch.from_numpy(X_train[b_idx:b_idx+BATCH_SIZE]).float()
			y_usage = torch.from_numpy(Y_train_usage[b_idx:b_idx+BATCH_SIZE]).float()

			if cuda:
				x = x.cuda()
				y_usage = y_usage.cuda()

			# Consume function, or encoder part
			pred_usage, h = model.consume(x)

			# Predict function, or decoder part
			preds = model.predict(pred_usage, h, WINDOW_TARGET_SIZE)

			# compute lose
			loss_usage = loss_fn(preds, y_usage)

			# backprop and update
			opt.zero_grad()

			loss_usage.backward()

			opt.step()

			total_usage_loss += loss_usage.item()


		train_loss.append(total_usage_loss)

		print("\tTRAINING: {} total train USAGE loss.\n".format(total_usage_loss))


		#################################################################################################################################################
		# TESTING

		# Everything is the exact same as testing, except we don't update the weights

		y_usage = None
		pred_usage = None
		preds = None

		total_usage_loss = 0

		all_preds = []

		for b_idx in range(0, X_test.shape[0], BATCH_SIZE):
			with torch.no_grad():

				x = torch.from_numpy(X_test[b_idx:b_idx+BATCH_SIZE])
				y_usage = torch.from_numpy(Y_test_usage[b_idx:b_idx+BATCH_SIZE])

				if cuda:
					x = x.cuda()
					y_usage = y_usage.cuda()

				pred_usage, h = model.consume(x)

				preds = model.predict(pred_usage, h, WINDOW_TARGET_SIZE)

				# compute loss
				loss_usage = loss_fn(preds, y_usage)

				total_usage_loss += loss_usage.item()

				if (epoch == epochs-1):
					all_preds.append(preds)


		test_loss.append(total_usage_loss)

		print("\tTESTING: {} total test USAGE loss".format(total_usage_loss))

		print("\tTESTING:\n")
		print("\tSample of prediction:")
		print("\t\t TARGET: {}".format(y_usage[-1].cpu().detach().numpy().flatten()))
		print("\t\t   PRED: {}\n\n".format(preds[-1].cpu().detach().numpy().flatten()))

		y_last_usage = y_usage[-1].cpu().detach().numpy().flatten()
		pred_last_usage = preds[-1].cpu().detach().numpy().flatten()

		t2_one_epoch = time.time()

		time_one_epoch = t2_one_epoch - t_one_epoch

		print("TIME OF ONE EPOCH: {} seconds and {} minutes".format(time_one_epoch, time_one_epoch/60.0))



	#################################################################################################################################################
	# PLOTTING

	# for plotting and accuracy
	preds = torch.cat(all_preds, 0)
	preds = preds.cpu().detach().numpy().flatten()

	actual = Y_test_usage.flatten()

	# for loss plotting
	train_loss_array = np.asarray(train_loss)
	test_loss_array = np.asarray(test_loss)

	len_loss = np.arange(len(train_loss_array))

	# unnormalizing
	preds_unnorm = (preds*std_usage) + mu_usage

	# computing Mean Absolute Error and Mean Absolute Percentage Error
	mae3 = (sum(abs(usage_actual - preds_unnorm)))/(len(usage_actual))
	mape3 = (sum(abs((usage_actual - preds_unnorm)/usage_actual)))/(len(usage_actual))

	print("\n\tACTUAL ACC. RESULTS: MAE, MAPE: {} and {}%".format(mae3, mape3*100.0))

	if to_plot:
		# plotting
		plt.figure(1)
		plt.plot(np.arange(len(preds)), preds, 'b', label='Predicted')
		plt.plot(np.arange(len(actual)), actual, 'g', label='Actual')
		plt.title("Predicted vs Actual, {} timesteps to {} timesteps".format(window_source_size, window_target_size))
		plt.xlabel("Time in 5 minute increments")
		plt.ylabel("Usage (normalized)")
		plt.legend(loc='lower left')

		plt.figure(2)
		plt.plot(np.arange(len(actual)), actual, 'g', label='Actual')
		plt.plot(np.arange(len(preds)), preds, 'b', label='Predicted')
		plt.title("Predicted vs Actual, {} timesteps to {} timesteps".format(window_source_size, window_target_size))
		plt.xlabel("Time in 5 minute increments")
		plt.ylabel("Usage (normalized)")
		plt.legend(loc='lower left')

		plt.figure(3)
		plt.plot(np.arange(len(y_last_usage)), y_last_usage, 'g', label='Actual')
		plt.plot(np.arange(len(pred_last_usage)), pred_last_usage, 'b', label='Predicted')
		plt.title("Predicted vs Actual last test example, {} timesteps to {} timesteps".format(window_source_size, window_target_size))
		plt.xlabel("Time in 5 minute increments")
		plt.ylabel("Usage (normalized)")
		plt.legend(loc='lower left')
		
		plt.figure(4)
		plt.plot(np.arange(len(usage_actual[-12*24*7:])), usage_actual[-12*24*7:], 'g', label='Actual')
		plt.plot(np.arange(len(preds_unnorm[-12*24*7:])), preds_unnorm[-12*24*7:], 'b', label='Predicted')
		plt.title("Predicted vs Actual: Case 2, Zoom last 7 days".format(window_source_size, window_target_size))
		plt.xlabel("Time in 5 minute increments")
		plt.ylabel("Usage (kW)")
		plt.legend(loc='lower left')

		plt.figure(5)
		plt.plot(len_loss, train_loss_array, 'k')
		plt.title("Train loss")
		plt.xlabel("Epoch")
		plt.ylabel("Loss")

		plt.figure(6)
		plt.plot(len_loss, test_loss_array, 'r')
		plt.title("Test Loss")
		plt.xlabel("Epoch")
		plt.ylabel("Loss")


		#plt.show()





	# total time of run
	t1 = time.time()
	total = t1-t0
	print("\nTIME ELAPSED: {} seconds OR {} minutes".format(total, total/60.0))

	#################################################################################################################################################
	# Save the initial model

	# Print model's state_dict
	print("Model 1's state_dict:")
	for param_tensor in model.state_dict():
		print(param_tensor, "\t", model.state_dict()[param_tensor].size())

	os.chdir('/Users/farewell/Desktop/forecasting_models')

	torch.save(model.state_dict(), "forecasting_model_meter_{}.pt".format(model_1))

	# when saving models built from multiple seeds,use:
	# torch.save(model.state_dict(), "forecasting_model_meter_{}_seed_{}.pt".format(model_1,seed))

	print("Save model 1 done.")
	os.chdir('/Users/farewell/Desktop/forecasting_set')

	#################################################################################################################################################


	print("\nEnd of run\n")


	for_plotting = [usage_actual, preds_unnorm, y_last_usage, pred_last_usage]


	if create_file:
		#################################################################################################################################################
		# PREDICTING ANOMALIES USING THRESHOLD

		# first threshold
		idxs_1 = []
		for i in range(len(usage_actual)):
			rng = usage_actual[i]*threshold_1
			low = usage_actual[i]-rng
			high = usage_actual[i]+rng
			if (preds_unnorm[i] < low or preds_unnorm[i] > high):
				idxs_1.append(i)

		print("TOTAL NUMBER OF ANOMALIES FOR 5% THRESHOLD: {}".format(len(idxs_1)))



		# second threshold
		idxs_2 = []
		for i in range(len(usage_actual)):
			rng = usage_actual[i]*threshold_2
			low = usage_actual[i]-rng
			high = usage_actual[i]+rng
			if (preds_unnorm[i] < low or preds_unnorm[i] > high):
				idxs_2.append(i)

		print("TOTAL NUMBER OF ANOMALIES FOR 10% THRESHOLD: {}".format(len(idxs_2)))




		# third threshold
		idxs_3 = []
		for i in range(len(usage_actual)):
			rng = usage_actual[i]*threshold_3
			low = usage_actual[i]-rng
			high = usage_actual[i]+rng
			if (preds_unnorm[i] < low or preds_unnorm[i] > high):
				idxs_3.append(i)

		print("TOTAL NUMBER OF ANOMALIES FOR 25% THRESHOLD: {}".format(len(idxs_3)))




		# fourth threshold
		idxs_4 = []
		for i in range(len(usage_actual)):
			rng = usage_actual[i]*threshold_4
			low = usage_actual[i]-rng
			high = usage_actual[i]+rng
			if (preds_unnorm[i] < low or preds_unnorm[i] > high):
				idxs_4.append(i)

		print("TOTAL NUMBER OF ANOMALIES FOR 50% THRESHOLD: {}\n".format(len(idxs_4)))






		# CREATING ANOMALIES CSV FILE
		  
		anom_data = pd.read_csv(filename)
		anom_data = anom_data[int(dataset.shape[0]*0.80):]
		anom_data = anom_data[WINDOW_SOURCE_SIZE:]
		anom_data = anom_data[:-remainder]
		anom_data = anom_data.reset_index(drop=True)
		anom_data['5p anomaly'] = 0
		anom_data['10p anomaly'] = 0
		anom_data['25p anomaly'] = 0
		anom_data['50p anomaly'] = 0

		for i in idxs_1:
			anom_data.at[i, '5p anomaly'] = 1
		for i in idxs_2:
			anom_data.at[i, '10p anomaly'] = 1 
		for i in idxs_3:
			anom_data.at[i, '25p anomaly'] = 1
		for i in idxs_4:
			anom_data.at[i, '50p anomaly'] = 1

		anom_data['Predicted_USG'] = preds_unnorm

		idx = anom_data.index.values
		anom_data['idx'] = idx

		# have to drop columns in order to keep old index

		output = anom_data[['DT_TIMESTAMP', 'year','month_of_yr', 'day_of_yr', 'day_of_month', 'day_of_week', 'weekend', 'holiday', 'hour', 'minute',
					   'season', 'P_USAGE','Predicted_USG',
				    '5p anomaly', '10p anomaly', '25p anomaly', '50p anomaly']]

		os.chdir('/Users/farewell/Desktop/forecasting_reports')

		output.to_csv(model_1 + "_MODEL_all.csv", sep=',', index=False)

		if just_anoms_file:
			output = output.loc[output['5p anomaly']==1]
			output.to_csv(model_1 + "_MODEL_anoms.csv", sep=',', index=False)
			print("Anomaly file saved as {}".format(model_1 + "_MODEL_anoms.csv"))

		print("All Data saved as {}".format(model_1+ "_MODEL_all.csv"))

		os.chdir('/Users/farewell/Desktop/forecasting_set')







	# return accuracy results, time, training and testing loss, and plotting arrays

	return mae3, mape3, total/60.0, train_loss, test_loss, for_plotting







################################################################################################################################

# This is where you can change the input parameters as mentioned above. Will re-copy to here as well:


# INPUTS:
#   seed:       Can be any number. Random number initializer. This is used to randomize all numbers, samples, etc. but to remember the
#               order of randomization to be able to reproduce results.
#   filename:   this is your filename, make sure it is in the same directory as the code or the path is set to where the file/dataset is
#   cuda:       2 options, either True or False. If True use GPU, if False use CPU
#   cell_type:  3 options, either 'rnn', 'gru', or 'lstm' -- this just chooses which cell to use in the model
#   window_source_size: Can be any reasonable number. Correlates to number of timesteps to use in consume function, before prediction.
#                       This is T, from paper.
#   window_target_size: Can be any reasonable number. Correlates to number of timesteps to use in prediction function. This is N, from paper.
#                       This is how many timesteps you would like to predict ahead.
#   epochs:     Can be any reasonable number. This tells the model how many iterations of training you would like. 10 epochs
#               was found to be a sufficient number of epochs. More can be used, but risk overfitting.
#   batch_size: Can be any reasonable number. This tells the model how many random samples to use at once. The higher this number is, the
#               faster training is, but the worse accuracy is (very slightly). The smaller it is, the slower training is, but better accuracy
#               (again very slightly). Recommend using a power of 2 as the batch size -- hence 32, 64, 128, or 256.
#   hs:         Can be any reasonable number. This tells the model how many hidden units to use in the respective cells. Try using powers of 2
#               here as well -- hence 32, 64, 128. The higher the hidden size, the slower the training, but usually slightly better accuracy
#   to_plot:        Can be either True or False. True will output plots, False will not
#   threshold_x : set thresholds as decimal. 0.05 is 5% and so forth
#   create_file: Can be either True or False. True will create anomalie file, False will not
#   just_anoms_file: Can be either True or False. If you would like the csv file with only the anomalies, keep True, otherwise keep False

os.chdir('/Users/farewell/Desktop/forecasting_set')


# for one dataset at a time:
if __name__ == "__main__":

	mae, mape, total_mins, train_loss, test_loss, for_plotting = main(seed=0, cuda=False,
		filename="121091.csv", cell_type='lstm', window_source_size=12, window_target_size=6, epochs=10,
		batch_size=256, hs=64, to_plot=False, threshold_1=0.05, threshold_2=0.10, threshold_3=0.25, threshold_4=0.50,
		create_file=True, just_anoms_file=True)






















