import numpy as np
import glob
from tqdm import tqdm
from ase.io import read,write
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.structure_prediction.volume_predictor import DLSVolumePredictor

lists = glob.glob('*.cif')
for ll in tqdm(lists):
	atoms = read(ll)

	struct = AseAtomsAdaptor.get_structure(atoms)
	v0 = struct.volume
	dls_predictor = DLSVolumePredictor()

	ref_struct = dls_predictor.get_predicted_structure(struct)
	v1 = ref_struct.volume
	name = ll.split('.')[0]+'_dlsVP.cif'
	ref_struct.to(filename=name)
	print(ll.split('.')[0],(v1/v0)**(1./3.))
