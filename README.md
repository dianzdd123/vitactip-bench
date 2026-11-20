# ViTacTip benchmark

ViTacTip combines a transparent skin with internal markers so that one camera can observe both the object surface and deformation during contact. This repository contains sensor-comparison notebooks and recognition training scripts associated with two ViTacTip publications.

## From the papers to the code

| Research question | Repository entry point | What the code measures |
| --- | --- | --- |
| How accurately can each sensor estimate an edge pose? | [Edge pose regression](Edge_Pose_Regression/) | One notebook each for TacTip, ViTac and ViTacTip. The targets are two translations and one rotation (`pose_1`, `pose_3`, `pose_6`). See the ICRA pose experiment and the journal paper’s Section V-C. |
| Can the contact point and force be inferred from a sensor image? | [Contact localisation and force regression](Force_Localisation_Regression/) | Three sensor-specific notebooks predict three position and three force components. The ICRA paper treats localisation and force estimation together. Section V-B of the journal paper benchmarks contact location. |
| How are marker positions represented? | [Marker detection](Marker_Detection/) | The Python utilities and example notebook extract blobs and construct marker graphs and Voronoi diagrams. These are tools for inspecting marker geometry. |
| How does a flat classifier compare with separate recognition heads? | [Recognition scripts](Recognition/) | A flat classifier and three variants of a two-head training script explore texture and stiffness recognition. The journal architecture in Fig. 6 also predicts material. |

The released files cover selected experiments. The grating-identification pipeline, the full three-head material/texture/hardness model and the LR-GAN and MR-GAN training pipelines are outside this release. The papers report the benchmark scores. This repository update did not rerun those experiments.

## Data and use

The original release links to the [ViTacTip evaluation dataset on Google Drive](https://drive.google.com/drive/folders/1VyEBp8wNfRd2YZ5skOsSgVkkF8hzOL7J?usp=drive_link). For regression, choose the notebook matching both the task and the sensor. Review its path configuration and point it to the corresponding images and label CSVs before running the training and evaluation cells. Keep each sensor paired with its own labels. For recognition, the expected directory layout and differences among the four scripts are described in the [recognition guide](Recognition/README.md).

## Publications

- Dandan Zhang, Wen Fan, Jialin Lin, Haoran Li, Qingzheng Cong, Weiru Liu, Nathan F. Lepora and Shan Luo. *Design and Benchmarking of a Multimodality Sensor for Robotic Manipulation With GAN-Based Cross-Modality Interpretation*. IEEE Transactions on Robotics, 2025. [DOI](https://doi.org/10.1109/TRO.2025.3526296)
- Wen Fan, Haoran Li, Weiyong Si, Shan Luo, Nathan F. Lepora and Dandan Zhang. *ViTacTip: Design and Verification of a Novel Biomimetic Physical Vision-Tactile Fusion Sensor*. IEEE International Conference on Robotics and Automation (ICRA), 2024. [DOI](https://doi.org/10.1109/ICRA57147.2024.10611186)
