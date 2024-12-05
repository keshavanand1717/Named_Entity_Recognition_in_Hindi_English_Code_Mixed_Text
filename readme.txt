
README

This guide will help you install the required packages for running the provided code.

Prerequisites
Make sure you have Python installed. The recommended version is Python 3.8 or later.

Installation Steps

1. Install Required Packages

   You can install all necessary packages at once by using pip with the following commands:

   ```
   pip install pandas numpy scikit-learn matplotlib torch torchcrf tensorflow keras gensim spacy sklearn-crfsuite
   ```

   Alternatively, you can install them individually:

   ```
   # General Data Processing
   pip install pandas numpy matplotlib

   # Machine Learning and Deep Learning
   pip install scikit-learn torch torchcrf tensorflow keras gensim

   # NLP
   pip install spacy sklearn-crfsuite
   ```

2. Verify Installation
   To verify that all packages are installed correctly, you can try importing them in a Python shell:

   ```python
   import pandas as pd
   import numpy as np
   import torch
   import torchcrf
   import tensorflow
   import keras
   import gensim
   import spacy
   import sklearn_crfsuite
   ```

3. Additional Notes
   - For CRF-based models, sklearn-crfsuite is used, which is a third-party library.
   - Some models, like torchcrf, require PyTorch. This command will install PyTorch's default version for your system configuration, but for the latest and optimized installation based on your hardware, follow the instructions on PyTorch’s official website: https://pytorch.org/
