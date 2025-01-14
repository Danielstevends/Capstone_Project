from setuptools import setup

setup(
    name="Generator_ML_Emission_Modeling",
    version="1.0",
    install_requires=[
        "pandas==1.5.3",
        "matplotlib==3.7.1",
        "openpyxl==3.1.2",
        "scikit-learn==1.2.2",
        "seaborn==0.12.2",
        "statsmodels==0.14.0",
        "xgboost==1.7.5",
    ],
    python_requires=">=3.11,<3.12",
    owner = 'Daniel Sitompul',
)
