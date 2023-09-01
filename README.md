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



<!-- ABOUT THE PROJECT -->
## About The Project


The Vaccine Adverse Event Reporting System (VAERS) contains
detailed reports of adverse events following vaccine administration.
However, efficiently and accurately searching for specific infor-
mation from VAERS poses significant challenges, especially for
medical experts. Natural language querying (NLQ) methods tackle
the challenge by translating the input questions into executable
queries, allowing for the exploration of complex databases with
large amounts of information. Most existing studies focus on the
relational database and solve the Text-to-SQL task. However, the
capability of full-text for Text-to-SQL is greatly limited by the data
structures and functionality of the SQL databases. In addition, the
potential of natural language querying has not been comprehen-
sively explored in the healthcare domain. To overcome these limita-
tions, we investigate the potential of NoSQL databases, specifically
Elasticsearch, and forge a new research direction for NLQ, which
we refer to as Text-to-ESQ generation. This exploration requires
us to re-design various aspects of NLQ, such as the target applica-
tion and the advantages of NoSQL database. In our approach, we
develop a two-stage controllable (TSC) framework consisting of
a question-to-question (Q2Q) translation module and an ESQ
condition extraction (ECE) module. These modules are carefully
designed to efficiently retrieve information from the VEARS data
stored in a NoSQL database. Additionally, we construct a dedicated
question-ESQ pair dataset called VAERSESQ, to support the task
in the healthcare domain. Extensive experiments were conducted
on the VAERSESQ dataset to evaluate the proposed methods. The
results, both quantitative and qualitative, demonstrate the accuracy
and efficiency of our approach in generating queries for NoSQL
databases, thus enabling efficient retrieval of VEARS data.


Use the `BLANK_README.md` to get started.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



### Built With

All the experiments were performed using NVIDIA Quadro RTX 5000 GPUs. The proposed TSC model
is implemented with PyTorch. We adopt the SGD with momentum optimizer during the training of the model parameters. The learning rate is set to 0.01. The experiments for all the models are
obtained by running 16 epochs with the mini-batch size 32. The development set is used to select the best model.

* M2M
* Bart
* LSTM
* RoBERTa
* DistilBERT
* RoBERTa+Bi-LSTM

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Getting Started

This is an example of how you may give instructions on setting up your project locally.
To get a local copy up and running follow these simple example steps.

### Prerequisites

All the experiments were performed using NVIDIA Quadro RTX 5000 GPUs. The proposed TSC model
is implemented with PyTorch. We adopt the SGD with momentum optimizer during the training of the model parameters. The learning rate is set to 0.01. The experiments for all the models are
obtained by running 16 epochs with the mini-batch size 32. The development set is used to select the best model.

### Installation

_Below is an example of how you can instruct your audience on installing and setting up your app. This template doesn't rely on any external dependencies or services._

1. Get a free API Key at [https://example.com](https://example.com)
2. Clone the repo
   ```sh
   git clone https://github.com/your_username_/Project-Name.git
   ```
3. Install NPM packages
   ```sh
   npm install
   ```
4. Enter your API in `config.js`
   ```js
   const API_KEY = 'ENTER YOUR API';
   ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage

Use this space to show useful examples of how a project can be used. Additional screenshots, code examples and demos work well in this space. You may also link to more resources.

_For more examples, please refer to the [Documentation](https://example.com)_

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ROADMAP -->
## Roadmap

- [x] Add Changelog
- [x] Add back to top links
- [ ] Add Additional Templates w/ Examples
- [ ] Add "components" document to easily copy & paste sections of the readme
- [ ] Multi-language Support
    - [ ] Chinese
    - [ ] Spanish

See the [open issues](https://github.com/othneildrew/Best-README-Template/issues) for a full list of proposed features (and known issues).

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTRIBUTING -->
## Contributing

Our major contributions can be summarized as follows

1. Formally propose and formulate the Text-to-ESQ task to support
NLQ on NoSQL database. To the best of our knowledge, this is
the initial comprehensive investigation of NLQ on the NoSQL
database
2. Propose a two-stage controllable (TSC) framework consisting of
two modules for Text-to-ESQ: (1) Question-to-question transla-
tion module for translating natural language questions into the
corresponding template questions, and (2) ESQ condition extrac-
tion module for parsing name entities about condition fields and
values from template questions for further populating the query
templates
3. Create a large-scale dataset VAERSESQ for Text-to-ESQ task for
retrieving information from VAERS data. For each question, we
include both template-based and natural language forms
4. Conduct an extensive experimental analysis of the VAERSESQ
dataset and demonstrate the effectiveness of the proposed two-
stage controllable method


<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACT -->
## Contact
Ping Wang -  pwang44@stevens.edu​

Wenlong Zhang -  wzhang71@stevens.edu

Project Link: (https://github.com/LEAF-Lab-Stevens/Text2ESQ​)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

Use this space to list resources you find helpful and would like to give credit to. I've included a few of my favorites to kick things off!

* [Choose an Open Source License](https://choosealicense.com)
* [GitHub Emoji Cheat Sheet](https://www.webpagefx.com/tools/emoji-cheat-sheet)
* [Malven's Flexbox Cheatsheet](https://flexbox.malven.co/)
* [Malven's Grid Cheatsheet](https://grid.malven.co/)
* [Img Shields](https://shields.io)
* [GitHub Pages](https://pages.github.com)
* [Font Awesome](https://fontawesome.com)
* [React Icons](https://react-icons.github.io/react-icons/search)

<p align="right">(<a href="#readme-top">back to top</a>)</p>
