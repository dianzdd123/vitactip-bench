# Texture and stiffness recognition

Section IV-F and Fig. 6 of the [2025 ViTacTip paper](https://doi.org/10.1109/TRO.2025.3526296) describe hierarchical recognition of material, texture and hardness. The four original scripts here support related comparisons. The multi-head generators discard the first field of the folder name and supervise two outputs, texture and stiffness. A material accuracy cannot be derived from their logs.

| Script | Experiment configured in its entry point |
| --- | --- |
| [Train_Tactile2_Benchmark-GoodNew.py](Train_Tactile2_Benchmark-GoodNew.py) | Flat classification. Each folder is a class and the model uses a 250-way output. |
| [Multi-Head-Hierarchy-CompareNew.py](Multi-Head-Hierarchy-CompareNew.py) | Parallel texture and stiffness heads. The fractional-data branch attempts to modify the Keras directory iterator directly. |
| [Multi-Head-Hierarchy-CompareNew3.py](Multi-Head-Hierarchy-CompareNew3.py) | Alternative fractional-data implementation using a custom batch generator. |
| [Multi-Head-Hierarchy-CompareNew3-finalupdate.py](Multi-Head-Hierarchy-CompareNew3-finalupdate.py) | Parallel-head sweep labelled 40%, 60% and 80%. The training loader sets a validation split but never selects a subset, so these runs use the full training directory. |

## Inputs and outputs

The scripts resolve `train_data/`, `val_data/` and `test_data/` relative to the working directory. Place RGB images inside class folders under each split. The multi-head loader expects exactly three underscore-separated fields, for example `train_data/sample_woven_soft/frame001.png`. Here `woven` and `soft` illustrate the two label positions. Validation and test labels must be present in the training label vocabulary. The flat classifier uses the entire folder name as the label and needs matching class indices across splits.

All four scripts compare CNN3, AlexNet, ResNet18, ResNet50, VGG16 and DenseNet121 using 256 × 256 images. The main blocks configure the sweep. Use separate working directories for variants because several output filenames overlap. Outputs include training histories, Keras model files and text summaries.

## Interpreting the original scripts

- The default multi-head configuration trains parallel heads. The scripts also define stiffness-first and texture-first builders. Before enabling stiffness-first, match targets by output name: the iterator yields texture first while this builder returns stiffness first.
- The final-update script selects GPU index 1 at startup. A CPU or single-GPU run requires adapting that device-selection line.
- For comparisons across training fractions, verify sample identities and counts together with both labels. The variants use different sampling implementations. For the flat and original parallel scripts, reduced-fraction runs depend on modifying iterator internals; compatibility depends on the Keras version.
- These historical scripts use the TensorFlow/Keras directory iterator API. No reference environment, classification split manifest or pretrained ViTacTip recognition weights accompany the scripts in this release.
