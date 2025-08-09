# [ACM CCS 2024] Moderator: Moderating Text-to-Image Diffusion Models through Fine-grained Context-based Policies

## 1-Introduction

We present Moderator, a policy-based model management system that allows administrators to specify fine-grained content moderation policies and modify the weights of a text-to-image (TTI) model to make it significantly more challenging for users to produce images that violate the policies. In contrast to existing general-purpose model editing techniques, which unlearn concepts without considering the associated contexts, Moderator allows admins to specify what content should be moderated, under which context, how it should be moderated, and why moderation is necessary. Given a set of policies, Moderator first prompts the original model to generate images that need to be moderated, then uses these self-generated images to reverse fine-tune the model to compute task vectors for moderation and finally negates the original model with the task vectors to decrease its performance in generating moderated content. We evaluated Moderator with 14 participants to play the role of admins and found they could quickly learn and author policies to pass unit tests in approximately 2.29 policy iterations. Our experiment with 32 stable diffusion users suggested that Moderator can prevent 65% of users from generating moderated content under 15 attempts and require the remaining users an average of 8.3 times more attempts to generate undesired content.

## 2-Prerequisite

Follow the instruction of [init.sh](init.sh) to install the environment.

```shell
bash init.sh
```

## 3-Run the test
```
python main.py
```

## 4-Reference
```
@inproceedings{10.1145/3658644.3690327,
    author = {Wang, Peiran and Li, Qiyu and Yu, Longxuan and Wang, Ziyao and Li, Ang and Jin, Haojian},
    title = {Moderator: Moderating Text-to-Image Diffusion Models through Fine-grained Context-based Policies},
    year = {2024},
    isbn = {9798400706363},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    url = {https://doi.org/10.1145/3658644.3690327},
    doi = {10.1145/3658644.3690327},
    booktitle = {Proceedings of the 2024 on ACM SIGSAC Conference on Computer and Communications Security},
    pages = {1181–1195},
    numpages = {15},
    keywords = {content moderation, policy language, text-to-image model},
    location = {Salt Lake City, UT, USA},
    series = {CCS '24}
}
``` 