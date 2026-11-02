# 모델과 데이터 출처

## 모델

- ViTPose `usyd-community/vitpose-base-simple`: [모델 카드](https://huggingface.co/usyd-community/vitpose-base-simple), Apache-2.0 표기. 가중치는 포함하지 않는다.
- RT-DETR `PekingU/rtdetr_r50vd_coco_o365`: [모델 카드](https://huggingface.co/PekingU/rtdetr_r50vd_coco_o365), Apache-2.0 표기. 가중치는 포함하지 않는다.
- `figures/model_card_excerpt.png`는 실습에 사용한 Hugging Face 모델 카드의 실제 화면 발췌다.

## COCO 데이터와 그림

COCO 이미지별 라이선스는 원본 `person_keypoints_val2017.json`의 `images[].license`와 `licenses[]`를 따른다. 이번에는 ID 4인 CC BY 2.0 이미지에 한정했다. 원본 메타데이터·출처와 해시는 `data/manifest.json`, 그림 출처는 `results/figure_credits.json`에 있다. COCO 주석 자체의 이용 조건은 [COCO Terms of Use](https://cocodataset.org/#termsofuse)를 참고한다.

| COCO 이미지 | 원본 사진 출처 | 라이선스 | 사용한 변경 |
|---|---|---|---|
| 33368 | [Flickr 원본 파일](http://farm5.staticflickr.com/4066/4287990820_0a3c8096ec_z.jpg), [원본 사진 식별 페이지](https://www.flickr.com/photo.gne?id=4287990820), [COCO 이미지](https://s3.amazonaws.com/images.cocodataset.org/val2017/000000033368.jpg) | [CC BY 2.0](https://creativecommons.org/licenses/by/2.0/) | 잘라낸 보기, 관절·골격 표시, 흐림·밝기 변형, affine crop 위 히트맵 겹침 |
| 574702 | [Flickr 원본 파일](http://farm3.staticflickr.com/2035/2252476376_bfc3cda192_z.jpg), [원본 사진 식별 페이지](https://www.flickr.com/photo.gne?id=2252476376), [COCO 이미지](https://s3.amazonaws.com/images.cocodataset.org/val2017/000000574702.jpg) | [CC BY 2.0](https://creativecommons.org/licenses/by/2.0/) | 잘라낸 보기, 관절·골격 표시, 흐림·밝기 변형 |

사진의 권리는 각 원저작자에게 있다. COCO에서 제공한 메타데이터에는 촬영자 이름이 없으며, 원본 출처와 사진 ID를 보존했다. 인물·사진 저작자가 이 실험을 보증하거나 참여한다는 뜻이 아니다. 원본 이미지 파일은 저장소에 포함하지 않는다. 사진이 포함된 파생 그림에도 원사진의 CC BY 2.0 출처 표기를 유지한다.

제공된 수업 PDF, 녹음 전사본, 개인 제출 보고서와 인증 자료는 재배포하지 않는다.
