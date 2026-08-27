# MeshFL [![Version](https://img.shields.io/badge/Version-2.0.0-brightgreen)]() [![MIT-License](https://img.shields.io/badge/license-MIT-green)](https://github.com/Mmasoud1/MeshFL/blob/main/LICENSE) [![PyTorch](https://img.shields.io/badge/PyTorch-Trained%20Model-blue)]() [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/18xjmBMR7EhuFVyhqhHuoGoCLJUuCwUA2?usp=sharing) [![JOSS](https://joss.theoj.org/papers/10.21105/joss.09357/status.svg)](https://doi.org/10.21105/joss.09357)

<div align="center">

**[Updates](#Updates) &emsp; [Doc](https://github.com/Mmasoud1/MeshFL/wiki/) &emsp; [News!](#News)**

</div>

<br>
 <img src="https://github.com/Mmasoud1/MeshFL/blob/main/css/logo/MeshFL.png"  width="25%" align="right">

  <p align="justify">
<b>MeshFL</b> is an advanced framework for distributed learning in neuroimaging. Built on the <a href="https://medium.com/pytorch/catalyst-neuro-a-3d-brain-segmentation-pipeline-for-mri-b1bb1109276a" target="_blank"  style="text-decoration: none"> MeshNet</a> models and <a href="https://developer.nvidia.com/flare" target="_blank"  style="text-decoration: none"> NVFlare</a>, it enables federated training for 3D MRI brain segmentation across decentralized data sites, maintaining privacy and efficiency.
 </p>

<p align="justify">
 For more information about MeshFL, please refer to this detailed <b><a href="https://github.com/Mmasoud1/MeshFL/wiki/"  style="text-decoration: none">Wiki</a></b>
</p>

<br>

## Key Features

* Federated training of the MeshNet model for 3D MRI brain segmentation.
* Supports decentralized learning across multiple sites using NVFlare.
* Automated data handling and splitting.
* Optimized GPU usage.
* Customizable training workflows with integrated Dice score evaluation.

<br>
<div align="center">

![Interface](https://github.com/Mmasoud1/MeshFL/blob/main/css/images/MeshFL_animated_output.gif)

**MeshFL training and MRI segmentation outputs**
</div>

## Getting Started
To start MeshFL, please refer to this steps <b><a href="https://github.com/Mmasoud1/MeshFL/wiki/Setup"  style="text-decoration: none">here</a></b>

## Demo

A complete demonstration of MeshFL training and inference is available:

[Demo Notebook](Examples/MeshFL_Demo.ipynb)

[Run on Colab](https://colab.research.google.com/drive/18xjmBMR7EhuFVyhqhHuoGoCLJUuCwUA2?usp=sharing)

This demo uses **150 training rounds** for faster execution. For improved performance, increasing training rounds (e.g., 300–350) can yield Dice scores above 0.93.

The demo includes:
- Training convergence visualization for 150 training rounds.
- Learning rate scheduling
- Final segmentation results

## Updates

* MeshFL <a href= "https://github.com/Mmasoud1/MeshFL/releases/tag/v2.0.0" target="_blank"  style="text-decoration: none"> v2.0.0 </a> has been released

## News!

* MeshFL [paper](https://joss.theoj.org/papers/10.21105/joss.09357) was published in the **Journal of Open Source Software (JOSS)** on **August 26, 2026**.

<div align="center">
   <a href="https://joss.theoj.org/papers/10.21105/joss.09357">
      <img src="https://github.com/Mmasoud1/HistoJS/blob/main/Demo/News/JOSS_Logo.png">
   </a>
</div>

<br>

## Citation

The MeshFL [paper](https://joss.theoj.org/papers/10.21105/joss.09357), corresponding to **MeshFL v2.0.0**, was published on **August 26, 2026** in the *Journal of Open Source Software (JOSS)*:

[![JOSS DOI](https://joss.theoj.org/papers/10.21105/joss.09357/status.svg)](https://doi.org/10.21105/joss.09357)

<br>

For **APA** style, please cite the paper as:

> Masoud, M., Reddy, P., Panta, S., & Plis, S. (2026). MeshFL: A Federated MeshNet Framework for 3D Brain MRI Segmentation. *Journal of Open Source Software, 11*(124), 9357. https://doi.org/10.21105/joss.09357

<br>

For **BibTeX**:

```bibtex
@article{Masoud2026,
  doi = {10.21105/joss.09357},
  url = {https://doi.org/10.21105/joss.09357},
  year = {2026},
  publisher = {The Open Journal},
  volume = {11},
  number = {124},
  pages = {9357},
  author = {Masoud, Mohamed and Reddy, Pratyush and Panta, Sandeep and Plis, Sergey},
  title = {MeshFL: A Federated MeshNet Framework for 3D Brain MRI Segmentation},
  journal = {Journal of Open Source Software}
}
```
<br>


## Contributions and Authorship Guidelines

We welcome contributions to MeshFL! Whether it's bug fixes, new features, or documentation improvements, feel free to submit an issue or a pull request.

If you modify or extend MeshFL in a derivative work intended for publication (such as a research paper or software tool), please cite and acknowledge the original MeshFL project and the original authors. 

We also request that significant contributions to derivative works be recognized by including original authors as co-authors, where appropriate.


## Acknowledgments

[NVFlare:](https://developer.nvidia.com/flare) Federated learning framework. 

[MeshNet:](https://medium.com/pytorch/catalyst-neuro-a-3d-brain-segmentation-pipeline-for-mri-b1bb1109276a) Volumetric dilated convolutional neural network architecture for MRI segmentation.

## Funding

MeshFL release V1.0.0 was funded by the NIH grant R01DA040487.    

<br />

<div align="center">

<img src='https://github.com/Mmasoud1/MeshFL/blob/main/css/logo/TReNDS_logo.jpg' width='300' height='100'></img>

</div>




