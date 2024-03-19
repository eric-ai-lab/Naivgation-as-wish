<div align="center">

<h1>Navigation as Attackers Wish? Towards Building Robust Embodied Agents under Federated Learning</h1>

<div>
    <a href='https://StylesZhang.github.io/' target='_blank'>Yunchao Zhang</a>,
    <a href='https://scholar.google.com/citations?user=5lFDxsMAAAAJ&hl=en&oi=ao' target='_blank'>Zonglin Di</a>,
    <a href='https://kevinz-01.github.io/' target='_blank'>Kaiwen Zhou</a>,
    <a href='https://cihangxie.github.io/' target='_blank'>Cihang Xie</a>,
    <a href='https://eric-xw.github.io/' target='_blank'>Xin Eric Wang</a>
</div>
<div>
    University of California, Santa Cruz, USA&emsp;
</div>

<h3><strong>Accepted to the Main Conference of <a href='https://2024.naacl.org/' target='_blank'>NAACL 2024</a></strong></h3>

<h3 align="center">
  <a href="https://arxiv.org/abs/2203.14936" target='_blank'>Paper |</a>
  <a href="https://styleszhang.github.io/pba/" target='_blank'>Project Page</a>
</h3>
</div>
<!--## Summary-->
<!--In this paper, we study an important and unique security problem in federated embodied AI -- whether the backdoor attack can manipulate the agent without influencing the performance and how to defend against the attack. We introduce a targeted backdoor attack NAW that successfully implants a backdoor into the agent and propose a promote-based defense framework PBA to defend against it.-->


We release the reproducible code here.

## Environment Installation

Python version: Need python 3.8
```
pip install -r python_requirements.txt
```

Please refer to [this link](https://github.com/peteanderson80/Matterport3DSimulator) to install Matterport3D simulator: 


## Pre-Computed Features
### ImageNet ResNet152

Download image features for environments for Envdrop model:
```
mkdir img_features
wget https://www.dropbox.com/s/o57kxh2mn5rkx4o/ResNet-152-imagenet.zip -P img_features/
cd img_features
unzip ResNet-152-imagenet.zip
```

### CLIP Features
Please download the CLIP-ViT features for CLIP-ViL models with this link:
```
wget https://nlp.cs.unc.edu/data/vln_clip/features/CLIP-ViT-B-32-views.tsv -P img_features
```

## Training and Testing On RxR

### Data
Please download the pre-processed data with link:
```
wget https://nlp.cs.unc.edu/data/vln_clip/RxR.zip -P tasks
unzip tasks/RxR.zip -d tasks/
```

### Testing NAW and PBA
For testing the performance of PBA on RxR dataset with model FedEnvDrop, please run

```
bash run/agent_rxr_envdrop_attack.bash
```

For testing the performance of PBA on RxR dataset with model FedCLIP-ViL, please run

```
bash run/agent_rxr_clip_vit_attack.bash
```
If you want to simply test NAW without any defense, please change the param  `defense_method` in the bash file from `PBA` to `mean`.    


## Training and Testing on R2R

### Download the Data
Download Room-to-Room navigation data:
```
bash ./tasks/R2R/data/download.sh
```

### Testing NAW and PBA
For testing the performance of PBA on R2R dataset with model FedEnvDrop, please run

```
bash run/agent_envdrop_attack.bash
```


For testing the performance of PBA on R2R dataset with model FedCLIP-ViL, please run

```
bash run/agent_clip_vit_attack.bash
```

If you want to simply test NAW without any defense, please change the param  `defense_method` in the bash file from `PBA` to `mean`.

## Related Links
- FedVLN: [paper](https://arxiv.org/abs/2203.14936), [code](https://github.com/eric-ai-lab/FedVLN)
- R2R Dataset: [paper](https://arxiv.org/pdf/1711.07280.pdf), [code](https://github.com/peteanderson80/Matterport3DSimulator)
- RxR Dataset: [paper](https://arxiv.org/abs/2010.07954), [code](https://github.com/google-research-datasets/RxR)

## Reference
If you use our work in your research or wish to refer to the baseline results published here, 
please use the following BibTeX entry. 


```shell 
@article{zhang2022navigation,
  title={Navigation as Attackers Wish? Towards Building Byzantine-Robust Embodied Agents under Federated Learning},
  author={Zhang, Yunchao and Di, Zonglin and Zhou, Kaiwen and Xie, Cihang and Wang, Xin Eric},
  journal={arXiv preprint arXiv:2211.14769},
  year={2022}
}
```
