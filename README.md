## Class Imbalance Redux in Python implementation

Requirements (Python 3):  `numpy`, `joblib` and `sklearn`

That's it. The class uses standard sklearn conventions.

#### Example of Usage

```python
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

clf = GradientBoostingClassifier(n_estimators = 3000)

cir = ClassImbalanceRedux(clf, n_bags = 10)
cir.fit(X_train, y_train) # y_train needs to be binary. np.int format only.
cir.predict(X_test)
cir.predict_proba(X_test)
cir.save('/home/user/model.npy')

## To load for production
cir = np.load('/home/user/model.npy').item()
```


#### Citation
```
@inproceedings{Wallace2011,
abstract = {Class imbalance (i.e., scenarios in which classes are unequally represented in the training data) occurs in many real-world learning tasks. Yet despite its practical importance, there is no established theory of class imbalance, and existing methods for handling it are therefore not well motivated. In this work, we approach the problem of imbalance from a probabilistic perspective, and from this vantage identify dataset characteristics (such as dimensionality, sparsity, etc.) that exacerbate the problem. Motivated by this theory, we advocate the approach of bagging an ensemble of classifiers induced over balanced bootstrap training samples, arguing that this strategy will often succeed where others fail. Thus in addition to providing a theoretical understanding of class imbalance, corroborated by our experiments on both simulated and real datasets, we provide practical guidance for the data mining practitioner working with imbalanced data.},
author = {Wallace, Byron C. and Small, Kevin and Brodley, Carla E. and Trikalinos, Thomas A.},
booktitle = {Proceedings - IEEE International Conference on Data Mining, ICDM},
doi = {10.1109/ICDM.2011.33},
isbn = {9780769544083},
issn = {15504786},
keywords = {Class imbalance,Classification},
pages = {754--763},
title = {{Class imbalance, redux}},
year = {2011}
}
```

