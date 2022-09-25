import numpy as np

line_dir = '/users/bdrechsl/jax_2/bdrechsl/IRAS16253/LineLists/'
H_list_file = line_dir + 'H_lines.npy'
H2_list_file = line_dir + 'H2_lines.npy'

H_list = np.load(H_list_file)
H2_list = np.load(H2_list_file)

line_dict = {'H': H_list, 'H2': H2_list}
