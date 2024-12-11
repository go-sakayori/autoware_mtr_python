# Motion Transformer (MTR)

> [Motion Transformer with Global Intention Localization and Local Movement Refinement](https://arxiv.org/abs/2209.13508)  
> [MTR++: Multi-Agent Motion Prediction with Symmetric Scene Modeling and Guided Intention Querying](https://arxiv.org/abs/2306.17770)

## Abstract

Predicting multimodal future behavior of traffic participants is essential for robotic vehicles to make safe decisions. Existing works explore to directly predict future trajectories based on latent features or utilize dense goal candidates to identify agent’s destinations, where the former strategy converges slowly since all motion modes are derived from the same feature while the latter strategy has efficiency issue since its performance highly relies on the density of goal candidates. In this paper, we propose the Motion TRansformer (MTR) framework that models motion prediction as the joint optimization of global intention localization and local movement refinement. Instead of using goal candidates, MTR incorporates spatial intention priors by adopting a small set of learnable motion query pairs. Each motion query pair takes charge of trajectory prediction and refinement for a specific motion mode, which stabilizes the training process and facilitates better multimodal predictions. Experiments show that MTR achieves state-of-the-art performance on both the marginal and joint motion prediction challenges, ranking 1st on the leaderboards of Waymo Open Motion Dataset.

## Setup

Before starting experiments, you need to build C++/CUDA extensions.

```shell
# Move to the MTR workspace
cd <AWML_PREDICTION_ROOT>/projects/MTR

# setup C++/CUDA extensions
python3 setup.py develop

# Back to the AWMLprediction workspace root
cd <AWML_PREDICTION_ROOT>
```

## Results

### Waymo

- Current time index: 10 (=1.1s)
- Predicted time length: 80 (=8.0s)

In order to prepare dataset for experiments, run the following command:

```shell
create-data waymo -i <INPUT_ROOT> -o <OUTPUT_ROOT> -t 91 -c 10
```

| Type       | mAP    | minADE | minFDE | MissRate |
| ---------- | ------ | ------ | ------ | -------- |
| Avg        | 0.3981 | 0.6168 | 1.2717 | 0.1419   |
| VEHICLE    | 0.4430 | 0.7593 | 1.5447 | 0.1542   |
| PEDESTRIAN | 0.4078 | 0.3529 | 0.7460 | 0.0819   |
| CYCLIST    | 0.3436 | 0.7382 | 1.5244 | 0.1897   |

## Citation

```latex
@article{shi2022motion,
  title={Motion transformer with global intention localization and local movement refinement},
  author={Shi, Shaoshuai and Jiang, Li and Dai, Dengxin and Schiele, Bernt},
  journal={Advances in Neural Information Processing Systems},
  year={2022}
}

@article{shi2023mtr,
  title={MTR++: Multi-Agent Motion Prediction with Symmetric Scene Modeling and Guided Intention Querying},
  author={Shi, Shaoshuai and Jiang, Li and Dai, Dengxin and Schiele, Bernt},
  journal={arXiv preprint arXiv:2306.17770},
  year={2023}
}

@article{shi2022mtra,
  title={MTR-A: 1st Place Solution for 2022 Waymo Open Dataset Challenge--Motion Prediction},
  author={Shi, Shaoshuai and Jiang, Li and Dai, Dengxin and Schiele, Bernt},
  journal={arXiv preprint arXiv:2209.10033},
  year={2022}
}
```
