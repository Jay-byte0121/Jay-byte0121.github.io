---
layout: post
title: a post with image galleries
date: 2024-12-04 01:59:00
description: this is what included image galleries could look like
tags: formatting images
categories: sample-posts
thumbnail: assets/img/9.jpg
images:
  lightbox2: true
  photoswipe: true
  spotlight: true
  venobox: true
---

# Intro
- 일반 [[ViT]] 는 pretraining → finetuning 둘다 classification task 로 진행
- DINO 에선 pretraining 단계를 SSL 로 하는 방법론 제시
- 성능도 더 좋고 내부 attn. map 의 feature 들이 공간적 정보를 담고 있다는 것을 보임
- 이 feature 들로 [[Dense Prediction]] tasks → sem. seg, depth estimation 가능 - [[Semantic Segmentation]], [[Depth Estimation]]

{% include figure.liquid loading="eager" path="/assets/img/dino-1.png" class="img-fluid rounded z-depth-1" %}

# Method
- Key ideas: [[Cross Entropy]], [[Multi Crop Strategy]], [[Momentum Teacher]], [[Centering]], [[Sharpening]]
- 기존 이미지를 [[Multi Crop Strategy]] 사용해서 2개의 global view, $V$ 개의 local view 로 augment
- Augmentations from [[BYOL]]
- Teacher 한테 global view, Student 한테 global view + local view 넘긴다음 CLS token [[Cross Entropy Loss]]

$$
P_s(x)^{(i)}=\frac{\exp \left(g_{\theta_s}(x)^{(i)} / \tau_s\right)}{\sum_{k=1}^K \exp \left(g_{\theta_s}(x)^{(k)} / \tau_s\right)}
$$

$$
\min _{\theta_s} \sum_{x \in\left\{x_1^g, x_2^g\right\}} \sum_{\substack{x^{\prime} \in V \\ x^{\prime} \neq x}} H\left(P_t(x), P_s\left(x^{\prime}\right)\right)
$$

- teacher softmax 하기 전 centering - collapse 방지용
	- centering 은 softmax temp. scaling 이랑 반대 효과

$$
c \leftarrow m c+(1-m) \frac{1}{B} \sum_{i=1}^B g_{\theta_t}\left(x_i\right),
$$

- output 은 [[Softmax Temperature Scaling]] 진행
- teacher 학습은 [[EMA]] 로
- 학습 완료 후:
	- Teacher model 로 inference.

{% include figure.liquid loading="eager" path="/assets/img/dino-2.png" class="img-fluid rounded z-depth-1" %}



