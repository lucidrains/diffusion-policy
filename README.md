## Diffusion Policy (wip)

Implementation of <a href="https://arxiv.org/abs/2303.04137">Diffusion Policy</a>, Toyota Research's supposed <a href="https://www.tri.global/news/toyota-research-institute-unveils-breakthrough-teaching-robots-new-behaviors">breakthrough</a> in leveraging DDPMs for learning policies for real-world Robotics

Update: Read the paper; it isn't really any new conceptual breakthrough. It simply applies popular text-to-image model architecture to policy generation. Observations are the text, and it then diffuses the actions. Feel free to correct me if I'm wrong in the discussions

## Citations

```bibtex
@article{Chi2023DiffusionPV,
    title   = {Diffusion Policy: Visuomotor Policy Learning via Action Diffusion},
    author  = {Cheng Chi and Siyuan Feng and Yilun Du and Zhenjia Xu and Eric A. Cousineau and Benjamin Burchfiel and Shuran Song},
    journal = {ArXiv},
    year    = {2023},
    volume  = {abs/2303.04137},
    url     = {https://api.semanticscholar.org/CorpusID:257378658}
}
```
