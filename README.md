# Zenseact Edge Annotationz Dataset

**The Zenseact Open Dataset referred to here was used during the [Edge Annotationz challenge](https://www.ai.se/en/data-factory/edge-annotationz-challenge). For the large-scale ZOD, please see [https://zod.zenseact.com/](https://zod.zenseact.com/).**

Zenseact Edge Annotationz Dataset is a sequential multimodal dataset consisting of 6666 unique sequences, captured by Zenseacts’ development vehicles in real-world traffic scenarios on the highway, country, and urban roads in and around Warsaw, Poland, during a three-week timespan. The data were collected during the day and night under varying weather conditions.
Each sequence is composed of three consecutive high-resolution (8MP) RGB camera images with 30 Hz frequency, in addition to the corresponding LiDAR, high-precision GPS (a.k.a OXTS), and the vehicle data sequences in the range [-1s, +1s] around the middle camera frame. Exhaustive annotations are provided for various autonomous driving tasks in [GeoJSON](https://geojson.org/) format for the middle frames of the sequences, allowing multi-task learning.

![DatasetOverviewTeaser](/assets/dataset_teaser.png)

Faces and the vehicle license plates are anonymized using the [brighterAI](https://brighter.ai/) tools for GDPR compliance and the intent to preserve every identity on roads. There are two anonymized RGB images provided per frame: One with blurred faces and license plates and the other with faces and license plates replaced by Deep Natural Anonymization. The technique is based on generative AI and leads to a minimal pixel impact. Information like line of sight of pedestrians is maintained and the anonymization method supports machine learning use cases in the best possible way. Having two different anonymized images per frame enables the impact study of varying anonymization techniques on the performance of the machine learning models.

More information about the dataset can be found [here](https://www.ai.se/sites/default/files/content/bilder/zenseact_dataset_intro.pdf).

# Dataset Structure
The Zenseact Edge Annotationz Dataset folder structure and the actual size of data in each folder is as follows.
[<img src="/assets/dataset_structure_graphical_condense.jpeg" width="400"/>](image.png)

Details about the content in each folder and the naming standard can be found [here](/assets/dataset_structure_details.jpeg).
# Getting started
The easiest way to start with the dataset is to use the provided functionalites in the development kit.
To get started, clone this repository, and use python 3.8.
Use the package manager [pip](https://pip.pypa.io/en/stable/#) to install the required packages listed in [requierment.txt](./requirements.txt).
This [notebook](./devkit_tutorial.ipynb) introduces working with different data loaders, coordinate transformation, and visualization functionalities, and this [notebook](./dataset_analysis.ipynb) can be used to analyze the dataset, and get a more detailed overview of its characteristics.
# License
**Dataset**: This dataset is the property of Zenseact AB (© 2021 Zenseact AB), and is licensed under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). Any public use, distribution or display of this dataset must contain this entire notice:
"For this dataset, Zenseact AB has taken all reasonable measures to remove all personal identifiable information, including faces and license plates. To the extent that you like to request removal of specific images from the dataset, please contact [privacy@zenseact.com](mailto:privacy@zenseact.com)".

**Development kit**: This development kit is the property of Zenseact AB (© 2022 Zenseact AB), and is licensed to under [MIT](https://opensource.org/licenses/MIT).

# Download
Zenseact Edge Annotationz Dataset is about 3.4TB in total when completely downloaded and extracted. Data from different sensors are split and can be downloaded separately as zip file(s); see this [section](#dataset-structure) for more details.

To download the Zenseact Edge Annotationz Dataset, please read the [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) Terms of Use of the license, and send your application to [opendataset@zenseact.com](mailto:opendataset@zenseact.com). Please specify your name, institution/organization, and an email connected to dropbox, in addition to a short description of how the dataset will be used. Scripts and examples of how to use the dataset can be found in this GitHub repository and are licensed under the terms of [MIT](https://opensource.org/licenses/MIT).

Moreover, the dataset is also available at AI Sweden Data Factory for all AI Sweden partners (see [this page](https://www.ai.se/en/data-factory/datasets/data-factory-datasets/zenseact-open-dataset)).

# Contributing
We welcome contributions to the data set or the development kit through pull requests, following the license terms. Please open an issue first to discuss what you would like to modify or add for major changes.

# Citation
If you publish work that uses Zenseact Edge Annotationz Dataset, please cite: [coming soon]()

# Team
The Zenseact Edge Annotationz Dataset is led by Mina Alibeigi. It is made possible by Daria Motorniuk, Oleksandr Panasenko, Jakub Bochynski, Dónal Scanlan, and Benny Nilsson. Special thanks to Jenny Widahl, Jonas Ekmark, Bolin Shao, Erik Rosén, and Erik Coelingh for their support, comments, and suggestions.

# Contact
For questions about the dataset, please [Contact Us](mailto:opendataset@zenseact.com).
