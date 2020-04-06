from setuptools import setup, find_packages

with open("README.md", "r") as readme_file:
    README = readme_file.read()

requirements = [
    "retry>=0.9.2",
    "pathos>=0.2.5",
    "nltk>=3.2.5",
    "textblob>=0.15.3",
    "pandas>=1.0.3",
    "python-dateutil>=2.8.1",
    "scikit-learn>=0.22.2",
    "gensim>=3.6.0"
]

classifiers = [
    "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    "Programming Language :: Python :: 3.6.9",
    "Operating System :: OS Independent", "Topic :: Covid 19 :: NLP",
    "Topic :: Utilities"
]

setup(name="c19",
      version="0.0.1",
      author="Atos Kaggle Crew",
      author_email="emeric.dynomant@gmail.com",
      description="Utilities lib for the following Kaggle notebook.",
      long_description=README,
      long_description_content_type="text/markdown",
      url="https://github.com/MrMimic/covid-19-kaggle",
      packages=find_packages(where="src/main/python/"),
      package_dir={"": "src/main/python/"},
      python_requires=">=3.6.9",
      install_requires=requirements,
      classifiers=classifiers)
