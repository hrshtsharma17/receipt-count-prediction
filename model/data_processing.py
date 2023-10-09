import numpy as np

def prepare_data(timeseries_data, n_steps):
	X, y = [],[]
	for i in range(len(timeseries_data)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the sequence
		if end_ix > len(timeseries_data)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = timeseries_data[i:end_ix], timeseries_data[end_ix]
		min_seq, max_seq = min(seq_x), max(seq_x)
		seq_x = (seq_x-min_seq)/(max_seq-min_seq)
		seq_y = (seq_y-min_seq)/(max_seq-min_seq)
		
		X.append(seq_x)
		y.append(seq_y)
		
	return np.array(X), np.array(y)

def prepare_last_sequence(timeseries_data, n_steps):
	seq_x = timeseries_data[-n_steps:]
	return np.array([seq_x])