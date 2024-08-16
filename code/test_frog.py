import tkinter as tk

from tkinter import *
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import mplcursors 

import math
activations = {
    'ReLU',
    'Sigmoid',
    'Tanh'
}


goodnesses = {
    'L2M_TK15',
    'L2M',
    'L1M',
    'L1M_TK15',
    'L2M_Split',
    'L2M_TK15_Split',
    'L1M_Split',
    'L1M_TK15_Split',
    'L2S',
    'L2S_TK15',
    'L1S',
    'L1S_TK15',
    'L2S_Split',
    'L2S_TK15_Split',
    'L1S_Split',
    'L1S_TK15_Split',
    'L2Sq',
    'L2Sq_TK15',
    'L2Sq_Split',
    'L2Sq_TK15_Split',
    'L2Mq',
    'L2Mq_TK15',
    'L2Mq_Split',
    'L2Mq_TK15_Split'
}


probabilities = [
    'SigmoidProbability_Theta_0',
    'SigmoidProbability_Theta_2',
    'SymmetricFFAProbability',
]

prob_mapper_0 = {
    'SigmoidProbability_Theta_0': 'Sigmoid_T0',
    'SigmoidProbability_Theta_2': 'Sigmoid_T2',
    'SymmetricFFAProbability': 'SFFA'
}

reverse_prob_mapper_0 = {
    'Sigmoid_T0': 'SigmoidProbability_Theta_0',
    'Sigmoid_T2': 'SigmoidProbability_Theta_2',
    'SFFA': 'SymmetricFFAProbability'
}

datasets = [
    'mnist',
    'emnist',
    'fashion',
    'kmnist',
    'cifar10'
]

color_list = [
    'blue',
    'red',
    'green',
    'black',
    'purple',
    'orange',
    'brown',
]

ATTRIBUTOS = ['Modelo', 'Accuracy', 'Hoyer', 'Separabilidad']

class ScatterplotApp:
    def __init__(self, root, dataset, names):
        self.root = root
        self.root.title("Scatterplot App")

        self.start = False
        self.names = names
        self.size = 10
        self.min_acc = 0.02
        
        self.current_color_scheme = "Base"
        self.color_funcs = {
            'Base': self.set_same_colors,
            'Activation': self.set_color_on_activation,
            'TopK': self.set_color_on_topk_used,
            'Norm': self.set_color_on_norm,
            'Split': self.set_color_on_split,
            'Probability': self.set_color_on_probability,
            'aggs' : self.set_color_on_L_vs_M
        }
    
        Grid.rowconfigure(root, 0, weight=1)
        Grid.columnconfigure(root, 0, weight=1)


        goodness_set = list(set([model[0].split('_')[1] for model in dataset]))
        self.filter_vars = {
            'activation' : dict([(name, tk.BooleanVar(value=True)) for name in activations]),
            'goodness' : dict([(name, tk.BooleanVar(value=True)) for name in goodness_set]),
            'split' : dict([(name, tk.BooleanVar(value=True)) for name in ['Split', 'No Split']]),
            'topk' : dict([(name, tk.BooleanVar(value=True)) for name in ['TopK15', 'No TopK15']]),
            'probability' : dict([(prob_mapper_0[name], tk.BooleanVar(value=True)) for name in probabilities]),
            'dataset' : dict([(name, tk.BooleanVar(value=True)) for name in datasets]),
            'fix_scale' : dict([('fix_scale', tk.BooleanVar(value=False))])
        }
        
        self.plot_frame = tk.Frame(self.root, highlightbackground="red", highlightthickness=6)
        self.plot_frame.grid(row=0, column=0, sticky="news")
        
        Grid.rowconfigure(self.plot_frame, 0, weight=1)
        Grid.columnconfigure(self.plot_frame, 0, weight=1)
        
        self.data = dataset
        self.filtered_model_names = [model[0] for model in self.data]

        self.add_axis_selectors()
        self.add_color_buttons()
        self.add_scatter_plot()
        
        self.filter_frame = tk.Frame(self.root, highlightbackground="blue", highlightthickness=2)
        self.filter_frame.grid(row=0, column=1, sticky="news")
        
    
        Grid.rowconfigure(self.filter_frame, 0, weight=1)
        Grid.columnconfigure(self.filter_frame, 0, weight=1)
        
        self.canvas1 = tk.Canvas(self.filter_frame)
        self.scrollbar = tk.Scrollbar(self.filter_frame, orient="vertical", command=self.canvas1.yview)
        self.scrollbar.pack(side="right", fill="y")
        
        self.canvas1.configure(yscrollcommand=self.scrollbar.set)
        self.canvas1.pack(side="left", fill="both", expand=True)
        
        self.checkbutton_frame = tk.Frame(self.canvas1)
        self.canvas1.create_window((0,0), window=self.checkbutton_frame, anchor='nw')
        
        for key, values in self.filter_vars.items():
            current_filter = tk.Frame(self.checkbutton_frame)
            current_filter.pack(pady=5)
            
            tk.Label(current_filter, text=key).pack(side='top')
            for name, var in values.items():
                tk.Checkbutton(current_filter, text=name, variable=var, command=self.update_plot).pack(side='left')


        current_filter = tk.Frame(self.checkbutton_frame)
        current_filter.pack(pady=5)
        self.size_slider = ttk.Scale(current_filter, from_=1, to=40, orient="horizontal", command=self.cont_update_size)
        self.size_label = tk.Label(current_filter, text=str(self.size_slider.get()))
        
        self.size_slider.bind("<ButtonRelease-1>", self.update_size)
        self.size_slider.set(10)
        self.size_slider.grid(row=0, column=0, sticky='w')
        
        self.size_label.grid(row=0, column=1, sticky='w')
        
        
        acccurrent_filter = tk.Frame(self.checkbutton_frame)
        acccurrent_filter.pack(pady=5)
        
        self.acc_slider = ttk.Scale(acccurrent_filter, from_=0, to=100, orient="horizontal", command=self.cont_update_acc)
        self.acc_slider.set(95)
        self.acc_label = tk.Label(acccurrent_filter, text=str(self.acc_slider.get()))
        
        self.acc_slider.bind("<ButtonRelease-1>", self.update_acc)
        self.acc_slider.set(10)
        self.acc_slider.grid(row=0, column=0, sticky='w')
        
        self.acc_label.grid(row=0, column=1, sticky='w')
        
        self.checkbutton_frame.update_idletasks()
        self.canvas1.config(scrollregion=self.canvas1.bbox("all"))
        
        self.start = True

        self.update_plot()

    def update_size(self, event):
        if not self.start:
            return
        self.size = int(math.floor(float(self.size_slider.get())))
        self.update_plot()
    
    def cont_update_size(self, event):
        if not self.start:
            return	
        self.size_label.config(text=f"{self.size_slider.get():.2f}")
    
    def cont_update_acc(self, event):
        if not self.start:
            return
        
        value = float(self.acc_slider.get())

        self.acc_label.config(text=f"{value:.2f}")
        
    def update_acc(self, event):
        if not self.start:
            return
        
        value = float(self.acc_slider.get())

        self.min_acc = value/100
        
        self.update_plot()

    def add_axis_selectors(self):
        variables = self.names

        
        self.x_var = tk.StringVar(self.plot_frame)
        self.y_var = tk.StringVar(self.plot_frame)
        self.x_var.set(variables[1])
        self.y_var.set(variables[2])
        
        x_dropdown = ttk.OptionMenu(self.plot_frame, self.x_var, *variables, command=self.update_plot)
        x_dropdown.pack(pady=10)
        y_dropdown = ttk.OptionMenu(self.plot_frame, self.y_var, *variables, command=self.update_plot)
        y_dropdown.pack(pady=5)
        
        self.x_var.set(variables[1])
        self.y_var.set(variables[2])

    def add_color_buttons(self):
        self.color_buttons_frame = tk.Frame(self.plot_frame)
        self.color_buttons_frame.pack(pady=5)

        ttk.Button(self.color_buttons_frame, text="Colores iguales", command=self.set_same_colors).pack(side=tk.LEFT, padx=5)
        ttk.Button(self.color_buttons_frame, text="Color por Activación", command=self.set_color_on_activation).pack(side=tk.LEFT, padx=5)
        ttk.Button(self.color_buttons_frame, text="Color por TopK", command=self.set_color_on_topk_used).pack(side=tk.LEFT, padx=5)
        ttk.Button(self.color_buttons_frame, text="Color por Norm", command=self.set_color_on_norm).pack(side=tk.LEFT, padx=5)
        ttk.Button(self.color_buttons_frame, text="Color por Prob", command=self.set_color_on_probability).pack(side=tk.LEFT, padx=5)
        ttk.Button(self.color_buttons_frame, text="Color por Split", command=self.set_color_on_split).pack(side=tk.LEFT, padx=5)
        ttk.Button(self.color_buttons_frame, text="Color por Agg", command=self.set_color_on_L_vs_M).pack(side=tk.LEFT, padx=5)
        ttk.Button(self.color_buttons_frame, text="Add Lines (Split)", command=self.connect_pair_with_lines).pack(side=tk.LEFT, padx=5)
        ttk.Button(self.color_buttons_frame, text="Add Lines (MeanSum)", command=self.connect_MS_pair_with_lines).pack(side=tk.LEFT, padx=5)
    
    
    
    def add_scatter_plot(self):
        self.figure = plt.Figure(figsize=(6, 6))
        self.ax = self.figure.add_subplot(111)
        self.scatter = self.ax.scatter([], [])
        self.ax.set_xlabel(self.x_var.get())
        self.ax.set_ylabel(self.y_var.get())
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        
    def get_filtered_model_names(self):
        valid_activations = [name+"_" for name, var in self.filter_vars['activation'].items() if var.get()]
        valid_goodnesses = [name for name, var in self.filter_vars['goodness'].items() if var.get()]
        valid_split = [name for name, var in self.filter_vars['split'].items() if var.get()]
        valid_topk = [name for name, var in self.filter_vars['topk'].items() if var.get()]
        valid_probabilities = [name for name, var in self.filter_vars['probability'].items() if var.get()]
        valid_datasets = [name for name, var in self.filter_vars['dataset'].items() if var.get()]
        
        temp_model_names = [models[0] for models in self.data]
        
        final_model_names = []
        for model in temp_model_names:
            activation = model.split('_')[0]+"_"
            goodness = model.split('_')[1]
            dataset = model.split('_')[-1]
            
            if self.data[temp_model_names.index(model)][1] < self.min_acc:
                continue
            
            if activation not in valid_activations:
                continue
            if dataset not in valid_datasets:
                continue
            if goodness not in valid_goodnesses:
                continue
            
            if 'Split' in model and 'Split' not in valid_split:
                continue
            if 'Split' not in model and 'No Split' not in valid_split:
                continue
            if 'TK15' in model and 'TopK15' not in valid_topk:
                continue
            if 'TK15' not in model and 'No TopK15' not in valid_topk:
                continue
            
            if "SymmetricFFAProbability" in model and "SFFA" not in valid_probabilities:
                continue
            if "SigmoidProbability_Theta_0" in model and "Sigmoid_T0" not in valid_probabilities:
                continue
            if "SigmoidProbability_Theta_2" in model and "Sigmoid_T2" not in valid_probabilities:
                continue
            
            final_model_names.append(model)
        
        self.filtered_model_names = final_model_names
        print("Recolected a total of ", len(final_model_names), " models")
                
        

    def update_plot(self, *args):
        
        self.get_filtered_model_names()

        fitlered_data = [model for model in self.data if model[0] in self.filtered_model_names]
        
        x_index = self.names.index(self.x_var.get())
        y_index = self.names.index(self.y_var.get())

        x_data = [row[x_index] for row in fitlered_data]
        y_data = [row[y_index] for row in fitlered_data]

        self.ax.clear()
        
        
        self.scatter = self.ax.scatter(x_data, y_data, c='blue', s=self.size) 
        self.ax.set_xlabel(self.x_var.get())
        self.ax.set_ylabel(self.y_var.get())
        
        if self.filter_vars['fix_scale']['fix_scale'].get():
            self.ax.set_xlim(0, 1)
            self.ax.set_ylim(0, 1)
        else:
            self.ax.autoscale()

        mplcursors.cursor(self.scatter).connect("add", lambda sel: sel.annotation.set_text(self.filtered_model_names[sel.target.index]))

        self.canvas.draw()
        
        self.color_funcs[self.current_color_scheme]()

    def connect_pair_with_lines(self):
        if not self.filter_vars['split']['Split'].get() or not self.filter_vars['split']['No Split'].get():
            return
        
        split_coords = []
        no_split_coords = []
        
        total_coords = []
        
        
        for model in self.data:
            if model[0] not in self.filtered_model_names:
                continue
                
            if "SymmetricFFAProbability" in model[0]:
                continue
            
            if 'Split' in model[0]:
                split_coords.append(model)
            else:
                no_split_coords.append(model)
        
        for i, split in enumerate(split_coords):
            pre, post = split[0].split('_Split_')[0], split[0].split('_Split_')[1]

            name_post = (pre+"_"+post).replace('Theta_0', 'Theta_2')
            for j, no_split in enumerate(no_split_coords):
                if name_post == no_split[0]:
                    total_coords.append((i, j))
                    break
        
        x_index = self.names.index(self.x_var.get())
        y_index = self.names.index(self.y_var.get())
        
        angles = []
        for pair in total_coords:
            self.ax.arrow(split_coords[pair[0]][x_index], split_coords[pair[0]][y_index], no_split_coords[pair[1]][x_index]-split_coords[pair[0]][x_index], no_split_coords[pair[1]][y_index]-split_coords[pair[0]][y_index], length_includes_head=True, width=0.0005, head_width=0.01, head_length=0.02, fc='gray', ec='gray')

            angle = math.degrees(math.atan2(no_split_coords[pair[1]][y_index]-split_coords[pair[0]][y_index], no_split_coords[pair[1]][x_index]-split_coords[pair[0]][x_index]))
            
            angle = math.radians(angle)
            
            angles.append(angle)
            
        self.ax_hist = self.figure.add_axes([0.2, 0.6, 0.2, 0.2], polar=True)
        self.ax_hist.hist(angles, bins=20, color='gray')
        self.ax_hist.set_title('Angle distribution (S-nS)')
        
        
        self.canvas.draw()

    def connect_MS_pair_with_lines(self):
        if not self.filter_vars['split']['Split'].get() or not self.filter_vars['split']['No Split'].get():
            return
        
        split_coords = []
        no_split_coords = []
        
        total_coords = []
        
        
        for model in self.data:
            if model[0] not in self.filtered_model_names:
                continue
                
            if "SymmetricFFAProbability" in model[0]:
                continue
            
            if 'L1M' in model[0] or 'L2M' in model[0]:
                split_coords.append(model)
            else:
                no_split_coords.append(model)
        
        for i, split in enumerate(split_coords):
            
            name_post = split[0].replace('L1M', 'L1S').replace('L2M', 'L2S')
            
            for j, no_split in enumerate(no_split_coords):
                if name_post == no_split[0]:
                    total_coords.append((i, j))
                    break
        
        x_index = self.names.index(self.x_var.get())
        y_index = self.names.index(self.y_var.get())
        
        angles = []
        for pair in total_coords:
            self.ax.arrow(split_coords[pair[0]][x_index], split_coords[pair[0]][y_index], no_split_coords[pair[1]][x_index]-split_coords[pair[0]][x_index], no_split_coords[pair[1]][y_index]-split_coords[pair[0]][y_index], length_includes_head=True, width=0.0005, head_width=0.01, head_length=0.02, fc='gray', ec='gray')

            angle = math.degrees(math.atan2(no_split_coords[pair[1]][y_index]-split_coords[pair[0]][y_index], no_split_coords[pair[1]][x_index]-split_coords[pair[0]][x_index]))
            
            angle = math.radians(angle)
            
            angles.append(angle)
            
        self.ax_hist = self.figure.add_axes([0.2, 0.6, 0.2, 0.2], polar=True)
        self.ax_hist.hist(angles, bins=20, color='gray')
        self.ax_hist.set_title('Angle distribution (L-M)')
        
        
        self.canvas.draw()    
        
    def set_same_colors(self):
        self.current_color_scheme = "Base"
        self.scatter.set_color('blue') 
        self.canvas.draw()

    def set_model_colors(self):
        self.current_color_scheme = "Model"
        self.scatter.set_color(['blue' if i % 2 == 0 else 'red' for i, _ in enumerate(self.data)]) 
        self.canvas.draw()
    
    def set_color_on_activation(self):
        self.current_color_scheme = "Activation"
        colores = []
        for model in self.filtered_model_names:
            activation = model.split('_')[0]
            mapping = {
                'ReLU': 'blue',
                'Sigmoid': 'red',
                'Tanh': 'green'
            }
            colores.append(mapping[activation])
        
        self.scatter.set_color(colores)
        self.canvas.draw()

    def set_color_on_norm(self):
        self.current_color_scheme = "Norm"
        colores = []
        for model in self.filtered_model_names:
            # Only L2
            if 'L2' in model and ('L2Sq' not in model and 'L2Mq' not in model):
                colores.append('blue')
            
            elif 'L1' in model:
                colores.append('red')
            
            elif 'L2Sq' in model or 'L2Mq' in model:
                colores.append('green')
            else:
                colores.append('black')
        
        self.ax.legend(handles=[plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='L2'),
                                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='L1'),
                                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='L2Sq/L2Mq'),
                                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=10, label='Other')], title='Norms')
        
        self.scatter.set_color(colores)
        self.canvas.draw()
    
    def set_color_on_L_vs_M(self):
        self.current_color_scheme = "aggs"
        colores = []
        for model in self.filtered_model_names:
            if 'L2M' in model or 'L1M' in model:
                colores.append('blue')
            else:
                colores.append('red')
                
        self.ax.legend(handles=[plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Mean'),
                                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Sum')], title='Aggregation') 
        
        self.scatter.set_color(colores)
        self.canvas.draw()
    
    def set_color_on_topk_used(self):
        self.current_color_scheme = "TopK"
        colores = []
        for model in self.filtered_model_names:
            if 'TK15' in model:
                colores.append('blue')
            else:
                colores.append('red')
        
        self.ax.legend(handles=[plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='TopK15'),
                                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='No TopK15')], title='TopK15') 
        
        self.scatter.set_color(colores)
        self.canvas.draw()
    
    def set_color_on_split(self):
        self.current_color_scheme = "Split"
        colores = []
        for model in self.filtered_model_names:
            if 'Split' in model:
                colores.append('blue')
            else:
                colores.append('red')
        
        self.ax.legend(handles=[plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Split'),
                                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='No Split')], title='Split')
        self.scatter.set_color(colores)
        self.canvas.draw()
    
    def set_color_on_probability(self):
        self.current_color_scheme = "Probability"
        colores = []
        for model in self.filtered_model_names:
            if 'SigmoidProbability' in model:
                colores.append('blue')
            else:
                colores.append('red')
        
        self.ax.legend(handles=[plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='SigmoidProbability'),
                                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='SymmetricFFAProbability')], title='Probabilities')
        
        
        
        
        self.scatter.set_color(colores)
        self.canvas.draw()

import json

with open('results/metrics/experimental_summary_test.json') as f:
    datapoints = json.load(f)


names = ['Model'] + datapoints['names']
data = datapoints['results']



data = [[key] + metrics for key, metrics in data.items()]
area_index = names.index('Convergence Area')

max_area = max([x[area_index] for x in data])

for i in range(len(data)):
    data[i][area_index] = data[i][area_index] / max_area

root = tk.Tk()
app = ScatterplotApp(root, data, names)
root.mainloop()
