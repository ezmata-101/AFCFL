conda activate flgpu
python fcfl_aa_main.py --fcfl=0 --fcfl_adaptive_alpha=false --fcfl_alpha_min=0.0 --fcfl_alpha_max=0.0 --fcfl_r=0.0
python fcfl_aa_main.py --fcfl=1 --fcfl_adaptive_alpha=false --fcfl_alpha_min=0.5 --fcfl_alpha_max=0.5 --fcfl_r=0.7
python fcfl_aa_main.py --fcfl=1 --fcfl_adaptive_alpha=true --fcfl_alpha_min=0.10 --fcfl_alpha_max=0.80 --fcfl_r=0.7


python fcfl_aa_main.py --iid=0 --fcfl=1 --fcfl_adaptive_alpha=0 --fcfl_adaptive_r=1
python fcfl_aa_main.py --iid=0 --fcfl=1 --fcfl_adaptive_alpha=1 --fcfl_adaptive_r=1
python fcfl_aa_main.py --iid=1 --fcfl=1 --fcfl_adaptive_alpha=0 --fcfl_adaptive_r=1
python fcfl_aa_main.py --iid=1 --fcfl=1 --fcfl_adaptive_alpha=1 --fcfl_adaptive_r=1