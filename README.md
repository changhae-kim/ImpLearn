# ImpLearn

This is an implementation of the importance learning algorithm, as presented in the manuscript jp-2024-05331h. For theoretical and historical backgrounds, you can also refer to https://doi.org/10.1039/C9RE00356H, https://doi.org/10.1063/5.0037450, and https://doi.org/10.1021/acs.jctc.3c00160.

The codes have been tested with the following versions of software, though older or newer versions could still be compatible:
```
python==3.10.4
ase==3.22.1
joblib==1.1.0
metric-learn==0.6.2
numpy==1.23.1
scikit-learn==1.1.1
scipy==1.8.1
```

The directory `ImpLearn` contains codes which define the classes and functions to be used in importance learning. This is a library which you can use to set up your own scripts for importance learning.

The directory `jp-2024-05331h` contains the scripts which were used to run the calculations in the manuscript jp-2024-05331h. This is an example which you can refer to when you want to run your own importance learning calculations.
