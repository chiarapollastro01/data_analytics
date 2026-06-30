| **Authors** | **Project** |  **Documentation** | **Build Status** | **Code Quality** | **Coverage** |
|:------------:|:-----------:|:------------------:|:----------------:|:----------------:|:------------:|
| [**Chiara Pollastro**](https://github.com/chiara01) <br/> S&C26 student | **data_analytics** | **TODO** | **Passing** | **TODO** | **TODO** |

<a href="https://www.unibo.it/">
  <div class="image">
    <img src="https://cdn.rawgit.com/physycom/templates/697b327d/logo_unibo.png" width="90" height="90">
  </div>
</a>

# data_analytics v0.1.0

## Shelter Animal Outcomes Analysis

This project explores the *Shelter Animal Outcomes* dataset to predict the outcome of animals entering a shelter (Adoption, Transfer, etc.). The goal is to build an end-to-end machine learning pipeline, from data cleaning and feature engineering to model evaluation.

* [Overview](#overview)
* [Prerequisites](#prerequisites)
* [Installation](#installation)
* [Usage](#usage)
* [Authors](#authors)
* [License](#license)

## Overview

This project implements a complete data science workflow to analyze animal shelter outcomes. It includes data preprocessing steps such as missing value imputation, outlier handling, and cyclic temporal feature engineering to improve model performance on imbalanced multi-class targets.

| :triangular_flag_on_post: Note |
|:-------------------------------|
| The raw datasets are not included in this repository due to their size. Please follow the instructions in the *Prerequisites* section to set them up. |

## Prerequisites

The complete list of requirements for this project is reported in the [requirements.txt](requirements.txt) file.

## Installation

Python version supported : ![Python version](https://img.shields.io/badge/python-3.10|3.11|3.12-blue.svg)

## Usage

The project is structured around the analisi_shelter.ipynb Jupyter Notebook. To get started:

* Download the train.csv dataset from the Kaggle Shelter Animal Outcomes competition.

* Create a data/ folder in the project root directory.

* Move the train.csv file into the data/ folder.

* Open the project in VS Code or Jupyter and execute the cells sequentially.

## Authors
* Chiara Pollastro

## License 
This project is licensed under the MIT License