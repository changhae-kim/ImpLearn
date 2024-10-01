# Phillips Catalyst

These are the scripts and data which were used to run the calculations in the manuscript jp-2024-05331h.

1. The subdirectory `slab_models` contains the atomistic silica slab models of Thompson and coworkers (https://doi.org/10.1021/acs.chemmater.1c04016).

2. The subdirectory `templates` contains the template geometries of the intermediates and transition states.

3. The script `preprocess.py` performs three tasks required prior to the importance learning iterations: (a) carve silanol pairs out of the slab models, (b) sort the silanol pairs into vicinal/non-vicinal/other, and (c) paste the template geometries onto the sites. Note that each task is implemented as a code block which interacts with the other tasks only through file read/write. Hence, you can run one task at a time by commenting out the other code blocks

4. The script `main.py` performs the importance learning iterations.

5. The script `tools.py` defines the functions to be used in `main.py`.

6. The script `imgfrq.py` can be used to detect Gaussian outputs with wrong numbers of imaginary frequencies.

## How to Run the Scripts

1. In order to run the scripts, you need to create a number of subdirectories:
```
mkdir \
    silanols \
    silanols_vicinal \
    silanols_nonvicinal \
    silanols_other \
    graft_vicinal \
    crest_vicinal \
    xtb_vicinal \
    gaussian_vicinal_wb97xd_tzvp_f321g \
    gaussian_vicinal_wb97xd_tzvp_f321g/bsse_sp \
    kernel_vicinal_wb97xd_tzvp_f321g_prep473K
```

2. Run the script `preprocess.py`:
```
python preprocess.py
```
The script should take several minutes to execute. As described earlier, you can divide the execution of this script into three stages, or run the entire script at once.

3. 





