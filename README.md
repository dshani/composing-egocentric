Requirements: micromamba

To set up micromamba environment run

'''
micromamba env create -f environment.yml
micromamba activate new_ego
pip install -r requirements.pip.txt
'''


To produce the paper figures run

'''
bash bash_scripts/paper_figs.sh
'''
