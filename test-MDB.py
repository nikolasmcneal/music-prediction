import matplotlib

matplotlib.use('Agg')
import os

#import hickle as hkl
import librosa
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import scipy
import soundfile as sf
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

from sklearn.metrics import mean_squared_error
from prednet import PredNet

# Visualization parameters
n_plot = 1 # number of plot to make (must be <= batch_size)

removenum = 0

nt = 27 #27 for RG

model_type = 'models/prednet-27-8-4layer-trained-on-train-set.pt'
folder = "medley_optim/"


class MyDataset(torch.utils.data.Dataset):
  def __init__(self, X, output_mode='error'):
    self.X = X
    self.output_mode = output_mode
    self.p_max = np.max(X)
    self.p_min = np.min(X)
    #print(np.min(self.X))
    #print(np.max(self.X))

  def preprocess(self, X):
    return (X.astype(np.float32) - self.p_min) / self.p_max

  def __len__(self):
    return len(self.X)

  def __getitem__(self, index):
    #print(self.X[index].shape)
    pad = np.zeros((nt, 128, 2))
    X = np.concatenate([pad, self.X[index], pad], axis=-1)
    X = self.preprocess(X)
    return X



filen = ""
# Model parameters
gating_mode = 'mul'
peephole = False
lstm_tied_bias = False
#batch_size = 10

A_channels = (1, 8, 16, 32)
R_channels = (1, 8, 16, 32)

# Load the testing data and perform preprocessing
data_prefix = "train_data"
# data_prefix = "/user_data/jjhuang1/test_data"
#rng = range(-1, 9)
rng = range(0, 1)
mses = []
mse_seq = []
idx = 0
for filename in sorted(os.listdir(data_prefix)):
	if filename.startswith("Medley-solos-DB_training"):
		for hint in rng:
			test_data = []
			num_files = 0
	#		for filename in os.listdir(data_prefix):

			if filename.endswith(".wav"): 
				file_location = "./" + data_prefix + "/" + filename
				num_files += 1
				y, sr = librosa.load(file_location, sr=None)
				filen = filename
				complete_melSpec = librosa.feature.melspectrogram(y=y, sr=sr, window=scipy.signal.hanning)
				complete_melSpec_db = librosa.power_to_db(complete_melSpec, ref=np.max)
				complete_melSpec_db_norm = (complete_melSpec_db * (255.0/80.0)) + 255.0
				complete_melSpec_db_norm_rot = np.rot90(complete_melSpec_db_norm.copy(),2)
				complete_melSpec_db_norm = torch.unsqueeze(torch.from_numpy(complete_melSpec_db_norm_rot.copy()),0)
				#complete_original = complete_melSpec_db_norm
				#print("Before splice: " + str(complete_melSpec_db_norm.shape))
				#print("Before splice dtype: " + str(complete_melSpec_db_norm.dtype))
				if hint >= 0:
					complete_melSpec_db_norm  = np.delete(complete_melSpec_db_norm, slice(0,hint), 2)
				if hint < 0:
					##if hint == -1:
					complete_melSpec_db_norm  = np.delete(complete_melSpec_db_norm, slice(-1+hint, -1), 2) #hacky, fix later
					#else:
				    
				#print("After splice: " + str(complete_melSpec_db_norm.shape))   
				#print("After splice dtype: " + str(complete_melSpec_db_norm.dtype))
				padmel = torch.zeros((1, 128, abs(hint)))
				if hint >= 0:
					complete_melSpec_db_norm = torch.cat((complete_melSpec_db_norm, padmel), dim=2)
				else:
					complete_melSpec_db_norm = torch.cat((padmel, complete_melSpec_db_norm), dim=2)
				#print("After concat: " + str(complete_melSpec_db_norm.shape))
				#print("After concat dtype: " + str(complete_melSpec_db_norm.dtype))
				#mellen = complete_melSpec_db_norm.shape[2]
				#complete_melSpec_db_norm = np.delete(complete_melSpec_db_norm, slice(mellen//2, mellen), 2)
				#print(str(complete_melSpec_db_norm.shape))
				#complete_melSpec_db_norm = np.delete(complete_melSpec_db_norm, numpy.s_[
				for j in range(1): #20
					curr = []
					curr_x = 0
					WINDOW_SIZE = 44
					SHIFT = 8
					for i in range(nt):#49):
						melSpec_db_norm = complete_melSpec_db_norm[0,:,curr_x:(curr_x+WINDOW_SIZE)].numpy()
						curr.append(melSpec_db_norm)
						curr_x += SHIFT
					if (len(curr) == nt): #49):
						test_data.append(np.asarray(curr))
				#print("num frame sequences:", len(test_data))
				test_dataset = MyDataset(test_data)
				test_loader_args = dict(shuffle = False, batch_size = 1, num_workers = 4, pin_memory = True)
				test_loader = DataLoader(test_dataset, **test_loader_args)
				#num_steps = 100
				input_size = (128, 48)

				model = PredNet(input_size, R_channels, A_channels, output_mode='prediction', gating_mode=gating_mode,
								peephole=peephole, lstm_tied_bias=lstm_tied_bias)
				model.load_state_dict(torch.load(model_type))
				# print('Model: ' + model_name)
				if torch.cuda.is_available():
					#print('Using GPU.')
					model.cuda()
				pred_MSE = 0.0
				copy_last_MSE = 0.0
				#target = np.zeros((1,), np.float32)
				for step, inputs in enumerate(test_loader): #for 5 times #for 100 times
					# ---------------------------- Test Loop ----------------------------
					inputs = inputs#.cuda() # batch x time_steps x channel x width x height
					new_shape = inputs.size() + (1,)
					inputs = inputs.view(new_shape)
					inputs = inputs.permute(0,1,4,2,3)
					pred = model(inputs) # (batch_size, channels, width, height, nt)
					pred = pred.cpu()
					pred = pred.permute(0,4,1,2,3) # (batch_size, nt, channels, width, height)
					#print(len(test_loader))				
					if step == len(test_loader)-1:
						inputs = inputs.detach().numpy() * 255.
						pred = pred.detach().numpy() * 255.

						inputs = np.transpose(inputs, (0, 1, 3, 4, 2))
						pred = np.transpose(pred, (0, 1, 3, 4, 2))
						
						inputs = inputs.astype(int)
						pred = pred.astype(int)
						plot_idx = np.random.permutation(inputs.shape[0])[:n_plot]

						aspect_ratio = float(pred.shape[2]) / pred.shape[3]
						fig = plt.figure(figsize = (nt, 2*aspect_ratio))
						gs = gridspec.GridSpec(2, nt)
						gs.update(wspace=0., hspace=0.)
						plot_save_dir = folder + "prediction_plots/"
						if not os.path.exists(plot_save_dir): os.mkdir(plot_save_dir)
						plot_idx = np.random.permutation(inputs.shape[0])[:n_plot]
												

						for i in plot_idx:
							for t in range(nt):
								#plt.subplot(gs[t])
								tmp = inputs[i,t]
								tmp = np.squeeze(tmp)
								direc = plot_save_dir + filename + "/" + str(removenum) + "/"
								if not os.path.exists(direc): os.makedirs(direc)
								plt.imsave(fname=direc + "orig" + str(t) + ".png", arr=tmp, cmap="gray", format="png")

								tmp2 = pred[i,t]
								tmp2 = np.squeeze(tmp2)
								if t == 1:
									predvals = tmp2[:,48-6]
								direc = plot_save_dir + filename + "/" + str(removenum) + "/"
								if not os.path.exists(direc): os.makedirs(direc)
								plt.imsave(fname=direc + "pred" + "-" + str(t) + ".png", arr=tmp2, cmap="gray", format="png")
								mse_curr = mean_squared_error(tmp, tmp2)
								mse_seq.append(mse_curr)
							print('filename: ' + str(filename) + "   pix shift: " + str(hint))
mses.append(mse_seq)
np.save(folder+'mses.npy', mses)
print(mses)

