# CGCNN-HD
The CGCNN-HD is a python code for dropout-based uncertainty quantification for stability prediction with CGCNN (by T. Xie et al.) developed by Jung group at KAIST.

Developers
----------
Juhwan Noh (jhwan@kaist.ac.kr)

Dependencies
------------
-  Python3
-  Numpy
-  Pytorch == 0.4.1.post2 (CUDA 8.0)
-  Pymatgen
-  Sklearn

How to use
------------
**1. Database setting**
- cd Mg-Mn-O_database   
- tar xvf lattice_scaled.tar   
- tar xvf org_cifs.tar   
- cp id_prop.r4_nn8.orgcif.csv org_cifs/id_prop.csv    
- cp atom_init.json org_cifs/    
- cp id_prop.r4_nn8.scaled.csv lattice_scaled/id_prop.csv    
- cp atom_init.json lattice_scaled/     

**2. Dropout Sampling**
> If you want to use org_cif database      
- Currently, crystal graph is constructed only if maximum number of neighboring atom = 8 and cutoff radius = 4A        
- Change root_dir = '/your/data/path/' in dropout_sampling.py to Mg-Mn-O_database/org_cifs/    
- python dropout_sampling.py cgcnn_hd_rcut4_nn8.best.pth.tar       
- You may get dropout_test.csv file (name,predicted mean,predicted standard deviation)         

> If you want to use scaled database                    
- Currently, crystal graph is constructed only if maximum number of neighboring atom = 8 and cutoff radius = 4A
- Change root_dir = '/your/data/path/' in dropout_sampling.py to Mg-Mn-O_database/lattice_scaled/    
- python dropout_sampling.py cgcnn_hd_rcut4_nn8.best.pth.tar       
- You may get dropout_test.csv file (name,predicted mean,predicted standard deviation)         

**3. Training with your own database**
- Currently, crystal graph is constructed only if maximum number of neighboring atom = 8 and cutoff radius = 4A              
- Change root_dir = '/your/data/path/' in model_train.py to path for your own dataset        
- Set MYPYTHON="your/python/path" and MODELPREF="your/model/pref" in training.sh with your own setting         
- sh training.sh         
