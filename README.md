# How to Evaluate and Refine your CAM

![ciao](./cover.png)

This repository contains the code, datasets, and resources associated with the paper "How to Evaluate and Refine your CAM"

## Installation

### Python venv

1) Clone the repository:

```bash
git clone https://github.com/liuktc/RefineCAM.git
cd RefineCAM
```

2) Create a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate
```

3) Install dependencies:

```bash
pip install -r requirements.txt
```

### Conda env

1) Clone the repository:

```bash
git clone https://github.com/liuktc/RefineCAM.git
cd RefineCAM
```

2) Create a conda environment:

```bash
conda env create -f environment.yml
```

3) Activate the environment:

```bash
conda activate myenv
```

## Usage

To run experiments on a single model and dataset, execute the following command:

```bash
python run_experiments.py --config_file ./configs/config.yaml
```

If you want to run experiments on different models or datasets, you can modify the config file to specify the desired models and datasets (see folder `configs` for more examples).


## License

This project is licensed under the MIT License – see the [LICENSE](https://github.com/liuktc/RefineCAM/blob/master/LICENSE) file for details.

You are free to use, modify, and distribute this code. If you use it for research, please cite the original paper.