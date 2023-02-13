[![DOI:10.1101/2021.10.05.463161](http://img.shields.io/badge/DOI-10.1101/2021.10.05.463161-B31B1B.svg)](https://doi.org/10.1101/2022.04.07.487434)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6793770.svg)](https://doi.org/10.5281/zenodo.6793770)


# experimental_macroecology



### Setting up your environment

You should be able to create a conda environment using a `environment.yml` file.

```bash
conda env create -f environment.yml
```

Sometimes this doesn't work. If so, below is the order of commands that I ran to build the conda environment.

```bash
conda create -n macroeco_phylo python=3.6
conda install -c etetoolkit ete3 ete_toolchain
conda install cython
pip install biom-format
conda install h5py
conda install matplotlib
conda install scipy
conda install pandas
```

Then activate your environment.

```bash
source activate macroecotools
```

### Getting the data

If you just want to regenerate our figures from the data we subsetted from the Earth Mirobiome Project, all you need is the data in our Zenodo repo, which should be placed in your file directory as `~/GitHub/macroeco_phylo/data`. 

If you want to repeat our subsetting procedure, you will need to download the EMP data. At the time that I started this study it was possible to obtain the first release of the EMP using File Transfer Protocol. Specificaly 

```bash
ftp://ftp.microbio.me/emp/release1/mapping_files/
ftp://ftp.microbio.me/emp/release1/otu_info/silva_123/
ftp://ftp.microbio.me/emp/release1/otu_tables/closed_ref_silva/
```

Alternatively, these files are also available on [Qiita](https://qiita.ucsd.edu/emp/).



### Running the analyses

```bash
python ~/GitHub/macroeco_phylo/scripts/run_analyses.py
```





For now, all data cleaning and analyses are in `Python/log_normal_analysis.py`. I'll change this structure at some point



### Python packages

- macroecotools
- scipy
- matplotlib
- numpy


conda activate macroecotools
