from setuptools import setup

setup(
    name='avocado',
    version='0.1.0',
    author='Jacob Schreiber',
    author_email='jmschr@cs.washington.edu',
    packages=['avocado'],
    license='LICENSE.txt',
    description='Avocado is a package for learning a latent representation of the human epigenome.',
    install_requires=[
        "IPython == 5.7.0",
        "pandas == 0.21.1",
        "numpy == 1.14.2",
        "keras == 2.0.8",
        "theano == 1.0.1",
        "scikit-learn == 0.19.1",
        "joblib == 0.11",
        "tqdm == 4.19.4",
        "shap",
        "xgboost == 0.7.post3",
        "matplotlib == 2.1.2",
        "seaborn == 0.8.1"
    ]
)
