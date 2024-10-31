# ImpLearn

This is an implementation of the importance learning algorithm, as presented in https://doi.org/10.1021/acs.jpcc.4c05331. For theoretical and historical backgrounds, you can also refer to earlier works of Peters and coworkers (https://doi.org/10.1039/C9RE00356H, https://doi.org/10.1063/5.0037450, and https://doi.org/10.1021/acs.jctc.3c00160).

## Directories

* The directory `ImpLearn` contains codes which define the classes and functions to be used in importance learning. This is a library which you can use to set up your own scripts for importance learning.

* The directory `jp-2024-05331h` contains the scripts which were used to run the calculations in https://doi.org/10.1021/acs.jpcc.4c05331. This is an example which you can refer to when you want to run your own importance learning calculations.

## Dependencies

The codes have been tested with these versions of software, though older or newer versions could be compatible:
```
dependencies:
  - crest==2.12
  - g16
  - python==3.10.4
  - xtb==6.6.0
  - pip:
    - ase==3.22.1
    - joblib==1.1.0
    - metric-learn==0.6.2
    - numpy==1.23.1
    - scikit-learn==1.1.1
    - scipy==1.8.1
```
