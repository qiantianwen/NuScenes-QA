# [AAAI 2024] NuScenes-QA

Official repository for the AAAI 2024 paper **[NuScenes-QA: A Multi-modal Visual Question Answering Benchmark for Autonomous Driving Scenario](https://arxiv.org/pdf/2305.14836.pdf)**.

![DataConstruction](docs/data_construction.png)

## :fire: News

- `2023.12.09`  Our paper is accepted by AAA! 2024! 
- `2023.09.04`  Our NuScenes-QA dataset v1.0 released.

## :hourglass_flowing_sand: To Do

- [x] Release question & anwswer data
- [ ] Release visual feature
- [ ] Release training and testing code

## :running: Getting Started

### Data Preparation

We have released our question-answer annotations, please download it from [HERE](https://drive.google.com/drive/folders/1jIkICT23wZWZYPrWCa0x-ubjpClSzOuU?usp=sharing).

For the visual data, you can download the origin nuScenes dataset from [HERE](https://www.nuscenes.org/download), and prepare the data refer to this [LINK](https://mmdetection3d.readthedocs.io/en/v0.16.0/datasets/nuscenes_det.html). As an alternative, you can also download our provided object-level features extracted using pre-trained detection models from [HERE]() (to be released soon).

### Training & Testing
Todo.

## :star: Others
If you have any questions about the dataset and its generation or the object-level feature extraction, feel free to cantact me with `twqian19@fudan.edu.cn`.


## :book: Citation
If you find our paper and project useful, please consider citing:
```bibtex
@article{qian2023nuscenes,
  title={NuScenes-QA: A Multi-modal Visual Question Answering Benchmark for Autonomous Driving Scenario},
  author={Qian, Tianwen and Chen, Jingjing and Zhuo, Linhai and Jiao, Yang and Jiang, Yu-Gang},
  journal={arXiv preprint arXiv:2305.14836},
  year={2023}
}
```