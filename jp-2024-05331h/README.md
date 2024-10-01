# Phillips Catalyst

These are the scripts and data which were used to run the calculations in the manuscript jp-2024-05331h.

## How to Run the Scripts

These are the step-by-step instructions to run the scripts and redo the calculations in the manuscript jp-2024-05331h, with some caveats to be specified below. The instructions assume that you are considering a Phillips catalyst grafted at 473.15 K.

1. Clone this repository to your home and then go into it:
```
git clone https://github.com/changhae-kim/ImpLearn
cd ImpLearn/jp-2024-05331h
```

2. Create the subdirectories:
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

3. Prepare the sites. The script can take several minutes to execute. As explained below, you can divide the execution of this script into three stages, or run the entire script at once.
```
python preprocess.py
```

4. Initiate the importance learning iteration. The script generates CREST inputs in the subdirectory `crest_vicinal`. Run these CREST jobs.
```
python main.py 473.15 0 -
```

5. Generate the xTB inputs. The script generates xTB inputs in the subdirectory `xTB_vicinal`. Run these xTB jobs.
```
python main.py 473.15 0 c
```

6. Generate the Gaussian inputs. The script generates Gaussian inputs in the subdirectory `gaussian_vicinal_wb97xd_tzvp_f321g`. Run these Gaussian jobs. For site-adsorbate complexes, compute the counterpoise corrections on the optimized geometries. Keep the inputs and outputs in the subsubdirectory `gaussian_vicinal_wb97xd_tzvp_f321g/bsse_sp`.
```
python main.py 473.15 0 cx
```

7. Train the ML models and then predict on the site ensemble. Based on the site-averaged rates, you can decide whether to do another iteration.
```
python main.py 473.15 0 cxgk
```

8. Initiate the next importance learning iteration. Then, repeat the above steps.
```
python main.py 473.15 1 -
```

## Caveats

In principle, you might expect to reproduce the "exact same" outputs as the manuscript jp-2024-05331h. Here, we are not concerned about the physics, which should be reproducible anytime anywhere, but we mean the computation: sampling the exact same set of sites in the exact same order. In practice, we expect your sampled sites to deviate after a few iterations. These are the reasons.

* The RNGs in the scripts use a hard-coded seed, so the scripts reproduce the same behaviors each time. However, CREST uses a new random seed each time, finding different candidate conformers.

* Due to the order-of-magnitude variations in the site-specific rates, the ML-predicted rates and sampling weights can be sensitive to the machine precision and architecture. For example, this code chooses different sites on computers with Intel vs. AMD CPUs. The differences arise the 15th decimal place at first but then propagate to more significant digits.

## Directories

* The subdirectory `slab_models` contains the atomistic silica slab models of Thompson and coworkers (https://doi.org/10.1021/acs.chemmater.1c04016).

* The subdirectory `templates` contains the template geometries of the intermediates and transition states.

* The script `preprocess.py` performs three tasks required prior to the importance learning iterations: (a) carve silanol pairs out of the slab models, (b) sort the silanol pairs into vicinal/non-vicinal/other, and (c) paste the template geometries onto the sites. Note that each task is implemented as a code block which interacts with the other tasks only through file read/write. Hence, you can run one task at a time by commenting out the other code blocks

* The script `main.py` performs the importance learning iterations. Because CREST, xTB, and Gaussian jobs can take a long time, the script is designed to generate the input files and then terminate. You can run these jobs on an HPC cluster, etc. Then, rerun the script to generate the next set of input files. You need to provide the script: (a) the grafting temperature in Kelvins, (b) the iteration number, and (c) the progress within the iteration. For example, suppose that you are considering a Phillips catalyst grafted at 473.15K, and you finished the xTB pre-optimization for the third iteration. Then, you can obtain the input files for the Gaussian optimization by executing: `python main.py 473.15 3 cx`.

* The script `tools.py` defines the functions to be used in `main.py`.

* The script `imgfrq.py` can be used to detect Gaussian outputs with wrong numbers of imaginary frequencies.
