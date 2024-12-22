# 비전 언어 모델(Vision Language Model)

## 1. VLM 활용

비전 언어 모델(VLM)은 이미지와 텍스트를 함께 처리해 이미지 캡셔닝, 시각적 질의 응답(visual question answering), 멀티모달 추론과 같은 작업을 가능하게 합니다. 

일반적인 VLM 구조는 시각적 특징을 추출하는 이미지 인코더, 시각적 표현과 텍스트 표현을 정렬하는 프로젝션 레이어, 그리고 텍스트를 처리하거나 생성하는 언어 모델로 구성됩니다. 이를 통해 모델은 시각적 요소와 언어적 개념 간의 연결을 형성할 수 있습니다.

VLM은 어떤 경우에 사용하는 지에 따라 다양한 구성으로 활용될 수 있습니다. 기본 모델은 일반적인 비전-언어 작업을 처리하며, 대화에 최적화된 변형 모델은 대화를 통한 상호작용을 지원합니다. 일부 모델은 예측을 시각적 근거에 기반하여 강화하거나 객체 탐지와 같은 특정 작업에 특화되기 위해 추가 구성 요소를 포함하기도 합니다.

VLM의 기술적인 세부사항과 활용법에 대한 더 많은 정보는 [VLM 활용](./vlm_usage.md) 페이지를 참조하세요.

## 2. VLM 파인튜닝

VLM을 파인튜닝한다는 것은 사전 학습된 모델을 특정 작업에 적합하도록 또는 특정 데이터셋에서 효과적으로 작동하도록 조정하는 과정을 의미합니다. 이 과정에서 모듈 1과 2에서 소개된 지도학습 파인튜닝(supervised fine-tuning), 선호 최적화(preference optimization), 또는 이 두 가지를 결합한 하이브리드 접근법과 같은 방법론을 적용할 수 있습니다.

LLM을 위한 핵심 도구와 기술은 VLM에서도 유사하게 적용되지만, VLM의 파인튜닝은 이미지 데이터의 표현과 준비에 추가적인 초점을 맞춰야 합니다. 이를 통해 모델이 시각적 데이터와 텍스트 데이터를 효과적으로 통합하고 처리하여 최적의 성능을 발휘할 수 있습니다. 데모 모델인 SmolVLM은 이전 모듈에서 사용된 언어 모델보다 훨씬 큰 모델이므로, 효율적인 파인튜닝 방법을 탐구하는 것이 중요합니다. 양자화(quantization)와 PEFT와 같은 기술을 활용하면 파인튜닝 과정을 더 접근 가능하고 비용 효율적으로 만들어, 더 많은 사용자가 모델을 실험할 수 있게 됩니다.

VLM 파인튜닝에 대한 자세한 가이드는 [VLM 파인튜닝](./vlm_finetuning.md) 페이지를 참조하세요.


## 실습 노트북


| 제목 | 설명 | 실습 내용 | 링크 | Colab |
|-------|-------------|----------|------|-------|
| VLM 활용 | 사전 학습된 VLM을 불러와서 다양한 작업에 활용하는 방법 배워보기 | 🐢 이미지 처리하기<br>🐕 배치 처리를 통해 여러 이미지를 한 번에 다루기 <br>🦁 동영상 처리하기| [Notebook](./notebooks/vlm_usage_sample.ipynb) | <a target="_blank" href="https://colab.research.google.com/github/huggingface/smol-course/blob/main/5_vision_language_models/notebooks/vlm_usage_sample.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> |
| VLM 파인튜닝 | 작업별 데이터셋에 맞춰 사전 학습된 VLM을 파인튜닝하는 방법 배워보기 | 🐢 기초적인 데이터셋을 사용해 파인튜닝하기<br>🐕 새로운 데이터셋 사용해보기<br>🦁 다양한 파인튜닝 방법 실험해보기 | [Notebook](./notebooks/vlm_sft_sample.ipynb)| <a target="_blank" href="https://colab.research.google.com/github/huggingface/smol-course/blob/main/5_vision_language_models/notebooks/vlm_sft_sample.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> | 


## 참고 
- [Hugging Face Learn: Supervised Fine-Tuning VLMs](https://huggingface.co/learn/cookbook/fine_tuning_vlm_trl)
- [Hugging Face Learn: Supervised Fine-Tuning SmolVLM](https://huggingface.co/learn/cookbook/fine_tuning_smol_vlm_sft_trl)  
- [Hugging Face Learn: Preference Optimization Fine-Tuning SmolVLM](https://huggingface.co/learn/cookbook/fine_tuning_vlm_dpo_smolvlm_instruct)  
- [Hugging Face Blog: Preference Optimization for VLMs](https://huggingface.co/blog/dpo_vlm)
- [Hugging Face Blog: Vision Language Models](https://huggingface.co/blog/vlms)
- [Hugging Face Blog: SmolVLM](https://huggingface.co/blog/smolvlm)  
- [Hugging Face Model: SmolVLM-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM-Instruct)
- [CLIP: Learning Transferable Visual Models from Natural Language Supervision](https://arxiv.org/abs/2103.00020)  
- [Align Before Fuse: Vision and Language Representation Learning with Momentum Distillation](https://arxiv.org/abs/2107.07651)  
