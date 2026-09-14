<p align="center">
  <a href="https://icpr2026.org/">
    <img src="https://icpr2026.org/Logos/icpr26Logo.svg" alt="ICPR 2026" height="55">
  </a>
  &nbsp;&nbsp;&nbsp;
  <a href="https://iapr-tc22-rrl.github.io/icpr2026/results/">
    <img src="https://iapr-tc22-rrl.github.io/assets/logoTC22.png" alt="IAPR TC22 Reproducibility" height="55">
  </a>
</p>

<h1 align="center">How to Evaluate and Refine your CAM</h1>

<p align="center">
  <b>Faithful CAM evaluation, a ground-truth benchmark, and high-resolution attribution maps.</b>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2605.14641">
    <img src="https://img.shields.io/badge/arXiv-2605.14641-B31B1B?style=for-the-badge&logo=arxiv&logoColor=white" alt="arXiv">
  </a>
  <a href="https://refinecam.github.io">
    <img src="https://img.shields.io/badge/Project-Website-6C63FF?style=for-the-badge" alt="Project Website">
  </a>
  <a href="https://iapr-tc22-rrl.github.io/icpr2026/results/">
    <img src="https://img.shields.io/badge/RRPR-Reproducibility%20Badge-18A558?style=for-the-badge" alt="Reproducibility Badge">
  </a>
</p>

<p align="center">
  <img src="./cover.png"
       alt="Overview of RefineCAM and CAM evaluation"
       width="100%">
</p>

<p align="center">
  <i>RefineCAM produces fine-grained attribution maps by combining information across multiple network layers.</i>
</p>

---

## Highlights

This work addresses two complementary problems in the evaluation and refinement of Class Attribution Maps (CAMs):

* **RefineCAM** combines CAMs across multiple network layers to produce higher-resolution and better-focused attribution maps.
* **ARCC** is a composite metric designed for more reliable evaluation of CAM explanations.
* **Synthetic CAM Benchmark** provides ground-truth attributions for systematically evaluating explanation metrics.

### RefineCAM and ARCC in `pytorch-grad-cam`

Implementations of both **RefineCAM** and **ARCC** are integrated into the widely used [`pytorch-grad-cam`](https://github.com/jacobgil/pytorch-grad-cam) library.

```bash id="3h7dz2"
pip install grad-cam
```

RefineCAM can be imported directly with:

```python id="4d6kom"
from pytorch_grad_cam import RefineCAM
```

ARCC is available as an evaluation metric:

```python id="a499y9"
from pytorch_grad_cam.metrics.ARCC import ARCC
```

For the maintained implementation and usage documentation, see [`pytorch-grad-cam`](https://github.com/jacobgil/pytorch-grad-cam).

## Paper

**How to Evaluate and Refine Your CAM**
Luca Domeniconi, Alessandra Stramiglio, Michele Lombardi, Samuele Salti

[Project Website](https://refinecam.github.io) ·
[arXiv](https://arxiv.org/abs/2605.14641) ·
[PDF](https://arxiv.org/pdf/2605.14641.pdf) ·
[pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam)

## Reproducibility

This work received the **ICPR 2026 Reproducible Research in Pattern Recognition (RRPR) Badge** following an independent reproducibility evaluation.

## Citation

If you use **RefineCAM**, **ARCC**, or the **synthetic benchmark**, please cite:

```bibtex id="h5179m"
@misc{2605.14641,
      Author = {Luca Domeniconi and Alessandra Stramiglio and Michele Lombardi and Samuele Salti},
      Title = {How to Evaluate and Refine your CAM},
      Year = {2026},
      Eprint = {arXiv:2605.14641},
}
```
