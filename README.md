## Description
Median housing value prediction

The housing data can be downloaded from <https://github.com/ageron/handson-ml/tree/master/datasets>.
The script has codes to download the data. We have modelled the median house value on given housing data.

# Installation
 **Using the Built Wheel File**
```bash
pip install dist/nonstandardcode-0.1-py3-none-any.whl
```
**Using the Source Distribution (.tar.gz)**
``` bash
pip install dist/nonstandardcode-0.1.tar.gz
```

# Installing Dependencies
Create and activate a Conda environment using:

```bash
conda env create -f conda/environment.yaml && conda activate mle-dev
```

# Verifying Installation
To check if the package is installed correctly, run:
```bash
python -c "import nonstandardcode; print('nonstandardcode installed successfully!')"
```

# Process to run the application

**Step-1:** Ingesting and pre-processing the housing dataset.

``` bash
python src/python_files/ingest_data.py --output-dir data/processed --log-level INFO --log-path logs/ingest_data.log
```

**step-2:** Train the model.
```bash
python src/python_files/train.py --train-path data/processed/train.csv --log-path logs/train.log
```

**step-3:** Evaluate and check the score of the model.
```bash
python src/python_files/score.py --model-path model/random_forest_model.pkl --test-path data/processed/val.csv --log-path logs/score.log
```

# Testing
Tests are written for various steps of the process. You can check if all the tests are running successfully using this command

```bash
pytest -v tests/
```
