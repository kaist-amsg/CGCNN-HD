import numpy as np
import glob
from tqdm import tqdm
from pymatgen import Structure

r_cut = 4.0
lists = glob.glob('*.cif')
for ll in tqdm(lists):
  atoms = Structure.from_file(ll)
  all_nbrs = atoms.get_all_neighbors(r_cut,include_index=True)
  num_of_nbrs = [len(nbrs) for nbrs in all_nbrs]
  if np.min(num_of_nbrs) >= 8:
	  print(','.join([ll.split('.')[0],str(-1.000)]))
