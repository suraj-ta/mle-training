## Description
Median housing value prediction

The housing data can be downloaded from <https://github.com/ageron/handson-ml/tree/master/datasets>.
The script has codes to download the data. We have modelled the median house value on given housing data.

# Installation
```bash
pip install -e .
```
# Process

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
