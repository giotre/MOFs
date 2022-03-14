# Sorption MOFs SL

This repository contains codes related to the publication "Minimal set of crystallographic descriptors for sorption properties in
hypothetical Metal Organic Frameworks and their role in sequential learning
optimization". Datasets and trained pipelines are published on our Zeonodo repository https://doi.org/10.5281/zenodo.6351366. 

In particular:
* Folder ```Models training + SHAP``` contains four ```.ipnyb``` files (one for each of the target properties of interest) to train a Random Forest based pipeline with hyperparameter tuning in 5-fold cross validation + SHAP analysis for detecting important features;
* Folder ```Sequental learning``` contains the code for running the SL (three ```.m``` files for Kriging, one ```.ipnyb``` file for Random-Forest- and COMBO-based methodologies);
* Folder ```Variable importances``` contains the complete ranking of the features, used to train the RF-regression models, provided by the SHAP analysis;
* File ```2D Map & Database optimum.ipynb``` contains the Database optimum in terms of the specific energy, its thermodynamic ideal cycle (Fig. 7 of the paper), and the comparative 2D Map (Fig. 6 of the paper).

## Datasets creation
We constructed the four MOFs datasets (published here https://doi.org/10.5281/zenodo.6351366), each one with the same 1557 features and a different target property among Henry coefficient for CO2 (column name ```'henry_coefficient_CO2_298K [mol/kg/bar]'```), working capacity for CO2 (column name ```'working_capacity_vacuum_swing_REPEAT_chg [mmol/g]'```), Henry coefficient for H2O (column name ```'henry_coefficient_H2O_298K [mol/kg/bar]'```) and surface area (column name ```'surface_area [m^2/g]'```), taking from here https://archive.materialscloud.org/2018.0016/v3
* the properties of interest, ```screening_data.tar.gz```, file ```top_MOFs_screening_data.csv``` containing over 8000 potential MOFs
* the descriptors from the featurization (see below) of the corresponding over 8000 CIF files among the 300000 in MOF_database.tar.gz

## Usage/Examples

To use one of the RF-pretrained models for doing new predictions:

* Featurize your Crystallographic Information Files (CIFs)

```python
import matminer
from matminer.featurizers.structure import JarvisCFID
import tqdm
import pandas as pd

cif_path = 'type the path of the folder containing your CIFs'
cif_files = os.listdir(cif_path) 
jarvis = JarvisCFID()

jarvis_features = []

for cif in tqdm.tqdm(cif_files):
    cif_struc = mp.Structure.from_file(cif_path + cif)
    cif_feature = jarvis.featurize(cif_struc)
    jarvis_features.append(cif_feature)

Matminer_labels = jarvis.feature_labels()
Data = pd.DataFrame(jarvis_features, index = cif_files, columns = Matminer_labels)
```
* Go on our Zeonodo https://doi.org/10.5281/zenodo.6351366 and download the model you are interested in, then importing the custom class for dropping the most correlated features, i.e.,

```python
from sklearn.base import TransformerMixin, BaseEstimator
from joblib import dump, load

class MyDecorrelator(BaseEstimator, TransformerMixin): 
    
    def __init__(self, threshold):
        self.threshold = threshold
        self.correlated_columns = None

    def fit(self, X, y=None):
        correlated_features = set()  
        X = pd.DataFrame(X)
        corr_matrix = X.corr()
        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if abs(corr_matrix.iloc[i, j]) > self.threshold: # we are interested in absolute coeff value
                    colname = corr_matrix.columns[i]  # getting the name of column
                    correlated_features.add(colname)
        self.correlated_features = correlated_features
        return self

    def transform(self, X, y=None, **kwargs):
        return (pd.DataFrame(X)).drop(labels=self.correlated_features, axis=1)

Henry_H2O_model = load('Henry_H2O_model.joblib')
```

* Predict with ```Henry_H2O_model.predict(Data)```

Otherwise, to use one of the AutoMatminer pretrained pipelines (supplementary material of the paper), download the one you are
interested in from our Zeonodo https://doi.org/10.5281/zenodo.6351366, and, after the featurization step, follow the instructions here https://hackingmaterials.lbl.gov/automatminer/basic.html#making-predictions.
