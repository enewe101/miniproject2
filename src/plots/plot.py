import os
import json 
import sys
import random
import numpy as np
import time 
import matplotlib.pyplot as plt
import matplotlib 
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FixedLocator, LinearLocator, FixedFormatter

RESULTS_DIR = 'results/'


class KNN_Data():

	FNAME = 'kNN/k_opt.json'

	def __init__(self):
		file_path = os.path.join(RESULTS_DIR, self.FNAME)
		self.data = json.loads(open(file_path).read())
		self.data.sort(None, lambda x: x['k'])

	
	def get_series(self):
		X = [d['k'] for d in self.data]
		Y = [d['accuracy'] for d in self.data]
		return X,Y


def plot_kNN():

	FNAME = 'figs/k_opt.pdf'
	plot = SimplePlot()
	data = KNN_Data()

	X,Y = data.get_series()
	plot.line(X,Y)
	plot.set_ylim(0.23, 0.57)
	plot.set_xlim(-2,32)
	plot.xlabel('accuracy')
	plot.ylabel('number of neighbors')
	plot.adjust(left=0.15)

	plot.save(os.path.join(RESULTS_DIR, FNAME))

def plot_NB():

	FNAME = 'figs/NB_representations.pdf'
	save_path = os.path.join(RESULTS_DIR, FNAME)

	# get a data manager and a plotter
	data = NB_Data()
	plot = SimplePlot(width=15)

	# plot each data series
	accuracies, run_params = data.get_ranked()
	Y = accuracies
	X = range(len(Y))
	plot.bar(X,Y)

	# add a legend and frame the plot nicely
	plot.set_xlim(-0.25, len(Y))
	plot.set_ylim(0.66, 0.86)
	plot.adjust()
	plot.hide_xticks()
	plot.ylabel('accuracy')

	# save the file
	plot.save(save_path)



def plot_LDA():

	FNAME = 'figs/LDA_num_features.pdf'
	save_path = os.path.join(RESULTS_DIR, FNAME)
	# get a data manager and a plotter
	data = LDA_Data()
	plot = SimplePlot()

	# plot each data series
	series_names = data.get_series_names()
	for series_name in series_names:
		X,Y = data.get_series(series_name)
		plot.line(X,Y,marker='.')

	# add a legend and frame the plot nicely
	series_names_no_underscore = [s.replace('_', ' ') for s in series_names]
	plot.add_legend(series_names_no_underscore)
	plot.shift_left(6)
	plot.set_ylim(0.58, 0.90)

	# save the file
	plot.save(save_path)



class NB_Data():

	REPRESENTATIONS = ['tf', 'tfidf', 'tficf', 'mod_tficf']

	def __init__(self):

		self.data = []
		for rep in self.REPRESENTATIONS:

			# the data for different representations is stored in different 
			# files.  Open the data for the current representation
			fname = '%s.json' % rep
			path = os.path.join(RESULTS_DIR, 'naive_bayes', fname)
			data_for_rep = json.loads(open(path).read())

			# When run with different representations, results were stored in
			# separate files.  Now we indicate within the data which 
			# representation it is, since we lose the connection to a file.
			for datum in data_for_rep:
				datum['params']['representation'] = rep

			# now add that data to the rest
			self.data.extend(data_for_rep)

	def get_ranked(self):
		self.data.sort(None, lambda x: x['results'])
		accuracies = [d['results'] for d in self.data]
		params = [d['params'] for d in self.data]
		return accuracies, params
		

	def get_data(self):
		return self.data



class SimplePlot():


	LINESTYLES = [
			{
				'linestyle': '-',
				'marker':'o',
				'markeredgecolor':'black',
				'markerfacecolor':'white', 
				'color':'black'

			}, {
				'linestyle': '-',
				'marker':'o',
				'markeredgecolor':'grey',
				'markerfacecolor':'grey', 
				'color':'grey'

			}, {
				'linestyle': '-.',
				'marker':'D',
				'markeredgecolor':'black',
				'markerfacecolor':'black', 
				'color':'black'

			}, {
				'linestyle': ':',
				'marker':'s',
				'markeredgecolor':'black',
				'markerfacecolor':'black', 
				'color':'black'

			}, {
				'linewidth': 2,
				'marker':'x',
				'markeredgecolor':'black',
				'markerfacecolor':'grey', 
				'color':'grey'
	
			}, {
				'linestyle': '--',
				'marker':'^',
				'markeredgecolor':'black',
				'markerfacecolor':'black', 
				'color':'black'
			}
	]


	def __init__(self, width=5, height=5):
		self.fig = plt.figure(figsize=(width, height))
		gs = gridspec.GridSpec(1,1)
		self.ax = plt.subplot(gs[0])
		self.series = []


	def bar(self, X, Y, **kwargs):

		# work out the bar styling
		defaults = {'width': 0.75, 'color':'0.25'}
		defaults.update(kwargs)

		# plot the bars
		self.series.append(self.ax.bar(X,Y,**defaults))

		# redraw the plat
		self.fig.canvas.draw()

		return self.series


	def line(self, X, Y, **kwargs):
		idx = len(self.series)
		length = len(self.LINESTYLES)
		style = self.LINESTYLES[idx % length]
		kwargs.update(style)
		self.series.append(self.ax.plot(X,Y,**kwargs))
		self.fig.canvas.draw()
		return self.series


	def hide_xticks(self):
		plt.setp(self.ax.get_xticklabels(), visible=False)
		self.fig.canvas.draw()


	def adjust(self, left=0.05, right=0.95, bottom=0.1, top=0.9):
		self.fig.subplots_adjust(
			left=left, right=right, bottom=bottom, top=top)
		self.draw()

	
	def draw(self):
		self.fig.canvas.draw()


	def ylabel(self, label, size=16):
		self.ax.set_xlabel(label, size=size)
		self.draw()


	def xlabel(self, label, size=16):
		self.ax.set_ylabel(label, size=size)
		self.draw()


	def get_fig(self):
		return self.fig


	def get_ax(self):
		self.ax


	def shift_down(self, shift_percent):
		min_y, max_y = plt.ylim()
		shift_by = shift_percent * (max_y - min_y) / float(100)
		plt.ylim(min_y - shift_by, max_y - shift_by)
		self.fig.canvas.draw()

	def shift_left(self, shift_percent):
		min_x, max_x = plt.xlim()
		shift_by = shift_percent * (max_x - min_x) / float(100)
		plt.xlim(min_x - shift_by, max_x - shift_by)
		self.fig.canvas.draw()

	def set_ylim(self, ymin, ymax):
		cur_y_min, cur_y_max = plt.ylim()

		if ymin is None:
			ymin = cur_y_min
		if ymax is None:
			ymax = cur_y_max

		plt.ylim(ymin, ymax)


	def set_xlim(self, xmin, xmax):
		cur_x_min, cur_x_max = plt.xlim()

		if xmin is None:
			xmin = cur_x_min
		if xmax is None:
			xmax = cur_x_max

		plt.xlim(xmin, xmax)


	def add_legend(self, names):
		label_objects = [s[0] for s in self.series]
		y_min, y_max = plt.ylim()
		x_min, x_max = plt.xlim()
		self.ax.legend(
			label_objects, 
			names, 
			loc=3,
			mode='expand',
			borderaxespad=0.,
			ncol=2,
			prop={'size':11},
			bbox_to_anchor=(0., 1.02, 1., 0.102)
		)
		self.fig.subplots_adjust(top=0.8)
		self.fig.canvas.draw()

	def save(self, path):
		self.fig.savefig(path)



class LDA_Data():

	FNAME = 'lda/performance.json'

	def __init__(self):

		# Find and load the data
		file_path = os.path.join(RESULTS_DIR, self.FNAME)
		unsorted_data = json.loads(open(file_path).read())

		# keep the data sorted
		self.data = sorted(
			unsorted_data, None, lambda x: ['num_features'], True)

		# we use the num_features datum for each LDA trial as an independant
		# variable
		self.num_features = [entry["num_features"] for entry in self.data]

		# get the names of the data series.  (`num_features` isn't a series 
		# (it's like a dependant variable, so filter it out)
		data_keys = sorted(self.data[-1].items(), None, lambda x: x[1], True)
		data_keys = [k[0] for k in data_keys]
		self.series_names = filter(lambda x: x != 'num_features', data_keys)


	def get_series_names(self):
		return self.series_names


	def get_series(self, series_name):
		performances = [entry[series_name] for entry in self.data]
		X,Y = self.num_features, performances
		return X,Y
		



class SimpleLinePlot():


	LINESTYLES = [
			{
				'linestyle': '-',
				'marker':'o',
				'markeredgecolor':'black',
				'markerfacecolor':'white', 
				'color':'black'

			}, {
				'linestyle': '-',
				'marker':'o',
				'markeredgecolor':'grey',
				'markerfacecolor':'grey', 
				'color':'grey'

			}, {
				'linestyle': '-.',
				'marker':'D',
				'markeredgecolor':'black',
				'markerfacecolor':'black', 
				'color':'black'

			}, {
				'linestyle': ':',
				'marker':'s',
				'markeredgecolor':'black',
				'markerfacecolor':'black', 
				'color':'black'

			}, {
				'linewidth': 2,
				'marker':'x',
				'markeredgecolor':'black',
				'markerfacecolor':'grey', 
				'color':'grey'
	
			}, {
				'linestyle': '--',
				'marker':'^',
				'markeredgecolor':'black',
				'markerfacecolor':'black', 
				'color':'black'
			}
	]


	def __init__(self, width=5, height=5):
		self.fig = plt.figure(figsize=(width, height))
		gs = gridspec.GridSpec(1,1)
		self.ax = plt.subplot(gs[0])
		self.series = []


	def plot(self, X, Y, **kwargs):
		idx = len(self.series)
		length = len(self.LINESTYLES)
		style = self.LINESTYLES[idx % length]
		kwargs.update(style)
		self.series.append(self.ax.plot(X,Y,**kwargs))
		self.fig.canvas.draw()
		return self.series


	def get_fig(self):
		return self.fig


	def get_ax(self):
		self.ax


	def shift_down(self, shift_percent):
		min_y, max_y = plt.ylim()
		shift_by = shift_percent * (max_y - min_y) / float(100)
		plt.ylim(min_y - shift_by, max_y - shift_by)
		self.fig.canvas.draw()

	def shift_left(self, shift_percent):
		min_x, max_x = plt.xlim()
		shift_by = shift_percent * (max_x - min_x) / float(100)
		plt.xlim(min_x - shift_by, max_x - shift_by)
		self.fig.canvas.draw()

	def set_ylim(self, ymin, ymax):
		cur_y_min, cur_y_max = plt.ylim()

		if ymin is None:
			ymin = cur_y_min
		if ymax is None:
			ymax = cur_y_max

		plt.ylim(ymin, ymax)

	def set_xlim(self, xmin, xmax):
		cur_x_min, cur_x_max = plt.xlim()

		if xmin is None:
			xmin = cur_x_min
		if xmax is None:
			xmax = cur_x_max

		plt.xlim(xmin, xmax)

	def add_legend(self, names):
		label_objects = [s[0] for s in self.series]
		y_min, y_max = plt.ylim()
		x_min, x_max = plt.xlim()
		self.ax.legend(
			label_objects, 
			names, 
			loc=3,
			mode='expand',
			borderaxespad=0.,
			ncol=2,
			prop={'size':11},
			bbox_to_anchor=(0., 1.02, 1., 0.102)
		)
		self.fig.subplots_adjust(top=0.8)
		self.fig.canvas.draw()

	def save(self, path):
		self.fig.savefig(path)


