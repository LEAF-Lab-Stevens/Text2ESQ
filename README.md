# Text-to-SQL Generation for Question Answering on Electronic Medical Records

[![image](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![image](https://img.shields.io/pypi/l/ansicolortags.svg)](https://github.com/wangpinggl/TREQS/blob/master/LICENSE)


- This repository is a two-stage controllable approach for efficient retrieval of Vaccine Adverse Events from the NoSQL database proposed in our ACMBCB'2023 paper:
[Text-to-ESQ: A Two-Stage Controllable Approach for Efficient Retrieval of Vaccine Adverse Events from NoSQL Database](https://dl.acm.org/doi/10.1145/3584371.3613008). 

- In this work, we are also releasing a large-scale VAERSESQ Dataset for Question-to-ESQ generation tasks in the healthcare domain. The VAERSESQ dataset is provided in [VAERSESQ dataset](https://github.com/LEAF-Lab-Stevens/Text2ESQ) in this repository. More details about VAERSESQ dataset are provided below.

- Links related to this work:
  - Paper: https://dl.acm.org/doi/10.1145/3584371.3613008
  - Dataset and codes: https://github.com/LEAF-Lab-Stevens/Text2ESQ
  - Slides: https://github.com/LEAF-Lab-Stevens/Text2ESQ/blob/WenlongZhang/Text-to-ESQ.pdf
  
- Updates (06/2024): We recently further explored some of the more complex questions and plan to release a new version of natural language questions. 

## Citation
Wenlong Zhang, Kangping Zeng, Xinming Yang, Tian Shi and Ping Wang. "Text-to-ESQ: A Two-Stage Controllable Approach for Efficient
Retrieval of Vaccine Adverse Events from NoSQL Database" In the 14th ACM Conference on Bioinformatics, 
Computational Biology, and Health Informatics, pp.1-10. 2023. 

```
@inproceedings{10.1145/3584371.3613008,
author = {Zhang, Wenlong and Zeng, Kangping and Yang, Xinming and Shi, Tian and Wang, Ping},
title = {Text-to-ESQ: A Two-Stage Controllable Approach for Efficient Retrieval of Vaccine Adverse Events from NoSQL Database},
year = {2023},
isbn = {9798400701269},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3584371.3613008},
doi = {10.1145/3584371.3613008},
abstract = {The Vaccine Adverse Event Reporting System (VAERS) contains detailed reports of adverse events following vaccine administration. However, efficiently and accurately searching for specific information from VAERS poses significant challenges, especially for medical experts. Natural language querying (NLQ) methods tackle the challenge by translating the input questions into executable queries, allowing for the exploration of complex databases with large amounts of information. Most existing studies focus on the relational database and solve the Text-to-SQL task. However, the capability of full-text for Text-to-SQL is greatly limited by the data structures and functionality of the SQL databases. In addition, the potential of natural language querying has not been comprehensively explored in the healthcare domain. To overcome these limitations, we investigate the potential of NoSQL databases, specifically Elasticsearch, and forge a new research direction for NLQ, which we refer to as Text-to-ESQ generation. This exploration requires us to re-design various aspects of NLQ, such as the target application and the advantages of NoSQL database. In our approach, we develop a two-stage controllable (TSC) framework consisting of a question-to-question (Q2Q) translation module and an ESQ condition extraction (ECE) module. These modules are carefully designed to efficiently retrieve information from the VEARS data stored in a NoSQL database. Additionally, we construct a dedicated question-ESQ pair dataset called VAERSESQ, to support the task in the healthcare domain. Extensive experiments were conducted on the VAERSESQ dataset to evaluate the proposed methods. The results, both quantitative and qualitative, demonstrate the accuracy and efficiency of our approach in generating queries for NoSQL databases, thus enabling efficient retrieval of VEARS data.},
booktitle = {Proceedings of the 14th ACM International Conference on Bioinformatics, Computational Biology, and Health Informatics},
articleno = {54},
numpages = {10},
keywords = {natural language querying, question translation, text-to-ESQ, VAERS, elasticsearch query},
location = {Houston, TX, USA},
series = {BCB '23}
}
```

## Dataset
VAERSESQ Dataset is created based on the publicly available [Vaccine Adverse Event Reporting System](https://vaers.hhs.gov/data.html) dataset.

- ```Template Questions:``` Based on the
information provided in the VAERS data and people’s interests in vaccine adverse events, we first summarized the possible asked questions on VAERS data and then normalized them as question templates by identifying medical entities in question and replacing them with generic holders
- ```Natural Questions:``` We utilized a state-of-the-art pre-trained language model, Many-to-Many multilingual translation model (M2M), and proposed a back-translation strategy to synthetic a group of natural language questions
- ```Elasticsearch Queries:```In general, each search template consists of two components, including a query template and its corresponding parameters. This template structure allows it to respond to various questions without changing the query template to a large extent.

- ```Example:``` Here we provide a data sample in MIMICSQL to illustrate the meaning of each element.

<img src="https://github.com/LEAF-Lab-Stevens/Text2ESQ/blob/WenlongZhang/QuestionSearchQuery.png" width="500px">

An example from VAERSESQ data. Colors show the correspondence between various components in the question template, NL/template questions, and query template.:
- `track_total _hits`: the query will retrieve all the relevant records.
- `field`: a specific field to search, which takes variables provided in the “params” section to query.
- `params`: provided variables section to query.
- `query`and `fuzziness`: designed for specific options to improve retrieval accuracy.

## Usage

- ```Training:``` python main.py 

- ```Validate:``` python main.py --task validate

- ```Test:``` python main.py --task test

## Evaluation

The codes for evaluation are provided in folder ```evaluation```. You can follow the following steps to evaluate the generated queries.

- Update the path to the MIMIC III data and generate MIMIC III database ```mimic.db``` with ```process_mimic_db.py```.

- Generate lookup table with ```build_lookup.ipynb```.

- Run overall evaluation or breakdown evaluation. If you plan to apply condition value recover technique, you need to run overall evaluation first (which will save the generated SQL queries with recovered condition values) before getting breakdown performance. Also, for evaluating on testing and development set, make sure to use the corresponding data file test.json and dev.json for testing and development sets, respectively.

## Results

Here we provide the results on the dataset.

<img src="https://github.com/LEAF-Lab-Stevens/Text2ESQ/blob/WenlongZhang/Results.png" width="800px">

<!-- Improved compatibility of back to top link: See: [https://github.com/othneildrew/Best-README-Template/pull/73](https://github.com/LEAF-Lab-Stevens/Text2ESQ/edit/main) -->
<a name="readme-top"></a>

<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->



<!-- PROJECT LOGO -->
<br />
<div align="center">

  <h3 align="center">Text-to-ESQ</h3>

</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>
