import os
import torch
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from natsort import natsorted
from tqdm import tqdm


dataset  = "MDB" #RG or MDB
h_index = 100
w_index = 20
sequences_num = 50


def processing(dataset, h_index, w_index):

  if dataset == "RG":
    typeneuron = "RG_optim/cell_states/"
  if dataset == "MDB":
    typeneuron = "medley_optim/cell_states/"


  direc = typeneuron
  files_list = []
  ext = '.pt' 
  file_dict = {}
  files = [i for i in natsorted(os.listdir(direc)) if os.path.splitext(i)[1] == ext]
  for i, f in enumerate(files):
    files_list.append(torch.load(direc + f, map_location=torch.device('cpu')).tolist())
  np_files_list = np.array(files_list)
  np_files_list = np_files_list.squeeze()


  np_files_list_new = []
  np_files_list_tmp = []

  mod = 0
  if dataset == "RG":
    mod = 49
  if dataset == "MDB":
    mod = 27

  for i, filename in tqdm(enumerate(np_files_list)):
    np_files_list_tmp.append(filename)
    if ((i+1) % mod) == 0:
      np_files_list_new.append(np_files_list_tmp)
      np_files_list_tmp = []
  print(len(np_files_list_new))

  all_sequences = []
  tmp_sequence = []
  for j in range(len(np_files_list_new)):
    for i in range(len(np_files_list_new[0])):
      tmp_sequence.append(np_files_list_new[j][i][h_index][w_index]) #sequence, frame, 128, 48
    all_sequences.append(tmp_sequence)
    tmp_sequence=[]


  return np_files_list_new, all_sequences

all_neurons_all_sequences, single_neuron_all_sequences= processing(dataset, h_index, w_index)



plot_direc = "plots/" + dataset+ "/"

print(sequences_num)
print(len(single_neuron_all_sequences))
for i in range(sequences_num):
  plt.plot(single_neuron_all_sequences[i])

plt.savefig(plot_direc + "{}_{}_sequences_neuron:_{}_{}".format(dataset, sequences_num, h_index, w_index))
plt.clf()
hist_single_neuron_all_sequences = [val for sublist in single_neuron_all_sequences for val in sublist]

plt.hist(hist_single_neuron_all_sequences, bins=200, range=(0,2))
plt.savefig(plot_direc + "{}_hist_sequences_neuron:_{}_{}".format(dataset, h_index, w_index))

