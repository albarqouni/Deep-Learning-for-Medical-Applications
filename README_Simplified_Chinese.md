# 在医学图像分析方向上深度学习论文清单
> 译者注：
> 本项目是一份关于医学图像分析上的论文清单,为了让更多的中国研究人员更好了解这一方向，我将该README文档翻译成简体中文。
向本项目的所有贡献者致敬。

> Translator's note:
 The project is the first list of deep learning papers on medical applications.In order to facilitate Chinese software developers to learn the deep learning on medical,I translated README file into simplified Chinese.
 Salute to all contributors to this project.
## 背景
据我们所知，这是第一份关于医学应用的深度学习论文清单。一般来说，有很多深度学习论文或计算机视觉的列表，例如[Awesome Deep Learning Papers](https://github.com/terryum/awesome-deep-learning-papers.git)。在此列表中，我尝试根据他们的深度学习技巧和学习方法对论文进行分类。我相信这份清单可能是一个很好的起点对于深度学习医学应用研究人员而言。

## 上榜标准
1. 2015年以来公开发布的顶尖深度学习论文。
2. 来自具有同行评议的期刊和知名会议。也包括一些最近发表在 arXiv的论文。
3. 深度学习技术，成像模态，临床数据库等元数据需要提供。

*期刊会议列表*

- **[医学图像分析 (MedIA)](https://www.journals.elsevier.com/medical-image-analysis/)**
- **[IEEE 医学图像学报 (IEEE-TMI)](https://ieee-tmi.org/)**
- **[IEEE 生物医学工程学报(IEEE-TBME)](http://tbme.embs.org/)**
- **[IEEE 生物医学与健康信息学杂志 (IEEE-JBHI)](http://jbhi.embs.org/)**
- **[国际计算机辅助放射学和外科学杂志 (IJCARS)](http://www.springer.com/medicine/radiology/journal/11548)**
- **医学影像信息处理国际会议 (IPMI)**
- **医学图像计算与计算机辅助干预国际会议 (MICCAI)**
- **计算机辅助干预信息处理国际会议 (IPCAI)**
- **IEEE国际生物医学成像研讨会 (ISBI)**

## 名词解释

*深度学习*
- NN(Neural Networks):神经网络
- MLP(Multilayer Perceptron): 多层感知机
- RBM(Restricted Boltzmann Machine): 受限玻尔兹曼机
- SAE(Stacked Auto-Encoders): 栈式自编码器
- CAE(Convolutional Auto-Encoders): 卷积的自编码器
- CNN(Convolutional Neural Networks): 卷积神经网络
- RNN(Recurrent Neural Networks): 循环神经网络
- LSTM(Long Short Term Memory): 长短期记忆神经网络
- MS-CNN(Multi-Scale/View/Stream CNN): 多尺度卷积神经网络
- MIL-CNN(Multi-instance Learning CNN): 多实例学习卷积神经网络
- FCN(Fully Convolutional Networks):全卷积神经网络

*医学名词*

- US(Ultrasound): 医学超声检查
- MR/MRI(Magnetic Resonance Imaging): 核磁共振成像
- PET(Positron Emission Tomography): 正子断层照影
- MG(Mammography): 乳房摄影术
- CT(Computed Tompgraphy): 计算机断层扫描
- H&E(Hematoxylin & Eosin Histology Images): 苏木精—伊红染色法


## 目录
### 深度学习
* [自编码器/ 栈式自编码器](#autoencoders--stacked-autoencoders)
* [卷积神经网络](#convolutional-neural-networks)
* [循环神经网络](#recurrent-neural-networks)
* [生成对抗网络](#generative-adversarial-networks)

### 医学应用
* [注解](#annotation)
* [分类](#classification)
* [检测](#detection--localization)
* [分割](#segmentation)
* [Registration](#registration)
* [回归](#regression)
* [图像重建](#https://arxiv.org/abs/1707.05927)
* [其他](#other-tasks)

## 文献索引

#### 研究类综述
| 计算机技术 | 医学技术 | 目标区域 | 标题| 数据库 | J/C | 年份 |
| ------ | ----------- | ----------- | ----------- |---|----------- | ---- |
| NN | H&E | 无 | Deep learning of feature representation with multiple instance learning for medical image analysis [[pdf]]() | | ICASSP| 2014|
| M-CNN   | H&E | 乳腺 | AggNet: Deep Learning From Crowds for Mitosis Detection in Breast Cancer Histology Images [[pdf]](http://ieeexplore.ieee.org/document/7405343/) | [AMIDA](amida13.isi.uu.nl)| IEEE-TMI | 2016| 
| FCN | H&E | 无 | Suggestive Annotation: A Deep Active Learning Framework for Biomedical Image Segmentation [pdf](https://arxiv.org/pdf/1706.04737.pdf)| | MICCAI | 2017 |
#### 分类问题

| 计算机技术 | 医学技术 | 目标区域 | 标题| 数据库 | J/C | 年份 |
| ------ | ----------- | ----------- | ----------- |---|----------- | ---- |
| M-CNN | CT | 肺 | Multi-scale Convolutional Neural Networks for Lung Nodule Classification [[pdf]](https://link.springer.com/chapter/10.1007/978-3-319-19992-4_46) | [LIDC-IDRI](https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI)| IPMI| 2015|
| 3D-CNN | MRI | 大脑 | Predicting Alzheimer's disease: a neuroimaging study with 3D convolutional neural networks [[pdf]](https://arxiv.org/abs/1502.02506) | [ADNI](adni.loni.usc.edu)| arXiv | 2015 |
| CNN+RNN| RGB | 眼睛 | Automatic Feature Learning to Grade Nuclear Cataracts Based on Deep Learning [[pdf]](https://pdfs.semanticscholar.org/2650/44769c0a35228d8512570f7ec4cc38e1c511.pdf)| | IEEE-TBME | 2015|
| CNN   | X-ray | 膝盖 | Quantifying Radiographic Knee Osteoarthritis Severity using Deep Convolutional Neural Networks [[pdf]](https://arxiv.org/pdf/1609.02469) | [O.E.1](https://oai.epi-ucsf.org/datarelease/)| arXiv | 2016|
| CNN | H&E | 甲状腺 | A Deep Semantic Mobile Application for Thyroid Cytopathology [[pdf]](http://proceedings.spiedigitallibrary.org/proceeding.aspx?articleid=2513164) | | SPIE | 2016|
| 3D-CNN, 3D-CAE | MRI | 大脑 | Alzheimer's Disease Diagnostics by a Deeply Supervised Adaptable 3D Convolutional Network [[pdf]](https://arxiv.org/pdf/1607.00556.pdf) | [ADNI](adni.loni.usc.edu) | arXiv| 2016
| M-CNN | RGB | 皮肤| Multi-resolution-tract CNN with hybrid pretrained and skin-lesion trained layers [[pdf]](http://www.cs.sfu.ca/~hamarneh/ecopy/miccai_mlmi2016a.pdf)|[Dermofit](https://licensing.eri.ed.ac.uk/i/software/dermofit-image-library.html)| MLMI | 2016|
| CNN | RGB | 皮肤,眼睛 | Towards Automated Melanoma Screening: Exploring Transfer Learning Schemes [[pdf]](https://arxiv.org/pdf/1609.01228.pdf)| [EDRA](http://dermoscopy.org/), [DRD](https://www.kaggle.com/c/diabetic-retinopathy-detection) | arXiv | 2016|
| M-CNN | CT | 肺 | Pulmonary Nodule Detection in CT Images: False Positive Reduction Using Multi-View Convolutional Networks [[pdf]](https://www.researchgate.net/profile/Geert_Litjens/publication/296624579_Pulmonary_Nodule_Detection_in_CT_Images_False_Positive_Reduction_Using_Multi-View_Convolutional_Networks/links/57f254cf08ae8da3ce517202.pdf) |  [LIDC-IDRI](https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI), [ANODE09](https://anode09.grand-challenge.org/), [DLCST](https://clinicaltrials.gov/ct2/show/study/NCT00496977) | IEEE-TMI | 2016|
| 3D-CNN | CT | 肺 | DeepLung: Deep 3D Dual Path Nets for Automated Pulmonary Nodule Detection and Classification [[pdf]](https://arxiv.org/abs/1801.09555) |  [LIDC-IDRI](https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI), [LUNA16](https://luna16.grand-challenge.org/) | IEEE-WACV | 2018|
| 3D-CNN | MRI | 大脑 | 3D Deep Learning for Multi-modal Imaging-Guided Survival Time Prediction of Brain Tumor Patients [[pdf]](http://www.unc.edu/~eadeli/publications/Dong_MICCAI2016.pdf)| |MICCAI | 2016|
| SAE | US, CT | 乳腺,大脑 | Computer-Aided Diagnosis with Deep Learning Architecture: Applications to Breast Lesions in US Images and Pulmonary Nodules in CT Scans [[pdf]](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4832199/pdf/srep24454.pdf) | [LIDC-IDRI](https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI)| Nature | 2016| 
| CAE | MG | 乳腺 |Unsupervised deep learning applied to breast density segmentation and mammographic risk scoring [[pdf]](https://ieeexplore.ieee.org/document/7412749/) | | IEEE-TMI | 2016| 
| MIL-CNN | MG | 乳腺 |Deep multi-instance networks with sparse label assignment for whole mammogram classification [[pdf]](https://link.springer.com/chapter/10.1007/978-3-319-66179-7_69) | [INbreast](http://medicalresearch.inescporto.pt/breastresearch/index.php/Get_INbreast_Database) | MICCAI | 2017| 
| GCN | MRI | 大脑 | Spectral Graph Convolutions for Population-based Disease Prediction [[pdf]](http://arxiv.org/abs/1703.03020) | [ADNI](http://adni.loni.usc.edu), [ABIDE](http://preprocessed-connectomes-project.org/abide/) | arXiv | 2017| 
| CNN | RGB | 皮肤 | Dermatologist-level classification of skin cancer with deep neural networks | | Nature | 2017 |
| FCN + CNN  | MRI | 肝，肝癌 | SurvivalNet: Predicting patient survival from diffusion weighted magnetic resonance images using cascaded fully convolutional and 3D convolutional neural networks [[pdf]](https://arxiv.org/abs/1702.05941) | | ISBI | 2017 |

#### 检测问题

| 计算机技术 | 医学技术 | 目标区域 | 标题| 数据库 | J/C | 年份 |
| ------ | ----------- | ----------- | ----------- |---|----------- | ---- |
| MLP   | CT | 头颈 | 3D Deep Learning for Efficient and Robust Landmark Detection in Volumetric Data [[pdf]](https://pdfs.semanticscholar.org/6cd3/ea4361e035969e6cf819422d0262f7c0a186.pdf) | | MICCAI | 2015|
| CNN   | US | 胎儿 | Standard Plane Localization in Fetal Ultrasound via Domain Transferred Deep Neural Networks [[pdf]](http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7090943) | | IEEE-JBHI | 2015|
| 2.5D-CNN   | MRI | 股骨 | Automated anatomical landmark detection ondistal femur surface using convolutional neural network [[pdf]](http://webpages.uncc.edu/~szhang16/paper/ISBI15_knee.pdf) | [OAI](https://oai.epi-ucsf.org/datarelease/)| ISBI | 2015|
| LSTM   | US | 胎儿 | Automatic Fetal Ultrasound Standard Plane Detection Using Knowledge Transferred Recurrent Neural Networks [[pdf]](https://link.springer.com/chapter/10.1007/978-3-319-24553-9_62) | | MICCAI | 2015|
| CNN   | X-ray, MRI | 头部 | Regressing Heatmaps for Multiple Landmark Localization using CNNs [[pdf]](https://www.tugraz.at/fileadmin/user_upload/Institute/ICG/Images/team_bischof/mib/paper_pdfs/MICCAI2016_CNNHeatmaps.pdf) | [DHADS](http://www.ipilab.org/BAAweb/)| MICCAI | 2016|
| CNN   | MRI, US, CT | - | An artificial agent for anatomical landmark detection in medical images [[pdf]](https://www5.informatik.uni-erlangen.de/Forschung/Publikationen/2016/Ghesu16-AAA.pdf) | [SATCOM](http://stacom.cardiacatlas.org/lv-landmark-detection-challenge/)| MICCAI | 2016|
| FCN   | US | 胎儿 | Real-time Standard Scan Plane Detection and Localisation in Fetal Ultrasound using Fully Convolutional Neural Networks [[pdf]](https://www.doc.ic.ac.uk/~bkainz/publications/Kainz_MICCAI2016b.pdf) | | MICCAI | 2016|
| CNN+LSTM   | MRI | 心脏 | Recognizing end-diastole and end-systole frames via deep temporal regression network [[pdf]](https://link.springer.com/chapter/10.1007/978-3-319-46726-9_31) | | MICCAI | 2016|
| M-CNN  | MRI | 心脏 | Improving Computer-Aided Detection Using Convolutional Neural Networks and Random View Aggregation Neural Networks [[pdf]](http://www.cs.jhu.edu/~lelu/publication/07279156.pdf) | | IEEE-TMI | 2016|
| CNN  | PET/CT | 心脏 | Automated detection of pulmonary nodules in PET/CT images: Ensemble false-positive reduction using a convolutional neural network technique Neural Networks [[pdf]](http://onlinelibrary.wiley.com/doi/10.1118/1.4948498/epdf) | | MP | 2016|
| 3D-CNN  | MRI | 大脑 | Automatic Detection of Cerebral Microbleeds From MR Images via 3D Convolutional Neural Networks [[pdf]](http://ieeexplore.ieee.org/document/7403984/#full-text-section) | | IEEE-TMI | 2016|
| CNN  | X-ray, MG | - | Self-Transfer Learning for Fully Weakly Supervised Lesion Localization [[pdf]](https://link.springer.com/chapter/10.1007/978-3-319-46723-8_28) | [NIH,China](http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=6663723), [DDSM,MIAS](http://www.mammoimage.org/databases/)  | MICCAI | 2016|
| CNN  | RGB | 眼睛 | Fast Convolutional Neural Network Training Using Selective Data Sampling: Application to Hemorrhage Detection in Color Fundus Images [[pdf]](http://ieeexplore.ieee.org/document/7401052/#full-text-section) | [DRD](https://www.kaggle.com/c/diabetic-retinopathy-detection/data), [MESSIDOR](http://www.adcis.net/en/Download-Third-Party/Messidor.html)  | MICCAI | 2016|
| GAN  | - | - | Unsupervised Anomaly Detection with Generative Adversarial Networks to Guide Marker Discovery | | IPMI | 2017|
| FCN | X-ray | 心脏 | CathNets: Detection and Single-View Depth Prediction of Catheter Electrodes | | MIAR | 2016|
| 3D-CNN | CT | 肺 | DeepLung: Deep 3D Dual Path Nets for Automated Pulmonary Nodule Detection and Classification [[pdf]](https://arxiv.org/abs/1801.09555) |  [LIDC-IDRI](https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI), [LUNA16](https://luna16.grand-challenge.org/) | IEEE-WACV | 2018|
| 3D-CNN | CT | 肺 | DeepEM: Deep 3D ConvNets with EM for weakly supervised pulmonary nodule detection [[pdf]](https://arxiv.org/abs/1805.05373v1) |  [LIDC-IDRI](https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI), [LUNA16](https://luna16.grand-challenge.org/) | MICCAI | 2018|

#### 分割问题
| 计算机技术 | 医学技术 | 目标区域 | 标题| 数据库 | J/C | 年份 |
| ------ | ----------- | ----------- | ----------- |---|----------- | ---- |
| U-Net  | - | - | U-net: Convolutional networks for biomedical image segmentation | | MICCAI | 2015|
| FCN   | MRI | 头颈 | Efficient multi-scale 3D CNN with fully connected CRF for accurate brain lesion segmentation [[pdf]](https://arxiv.org/pdf/1603.05959) | | arXiv | 2016 |
| U-Net   | CT | 头颈 | AnatomyNet: Deep learning for fast and fully automated whole‐volume segmentation of head and neck anatomy [[pdf]](https://www.researchgate.net/profile/Wentao_Zhu4/publication/329224429_AnatomyNet_Deep_Learning_for_Fast_and_Fully_Automated_Whole-volume_Segmentation_of_Head_and_Neck_Anatomy/links/5c075ae4458515ae5447b0eb/AnatomyNet-Deep-Learning-for-Fast-and-Fully-Automated-Whole-volume-Segmentation-of-Head-and-Neck-Anatomy.pdf) | | Medical Physics | 2018 |
| FCN   | CT | 肝，肝癌 | Automatic Liver and Lesion Segmentation in CT Using Cascaded Fully Convolutional Neural Networks and 3D Conditional Random Fields  [[pdf]](https://arxiv.org/abs/1610.02177) | | MICCAI | 2016 |
| 3D-CNN | MRI | 脊柱 | Model-Based Segmentation of Vertebral Bodies from MR Images with 3D CNNs | | MICCAI | 2016 |
| FCN   | CT | 肝，肝癌 | Automatic Liver and Tumor Segmentation of CT and MRI Volumes using Cascaded Fully Convolutional Neural Networks [[pdf]](https://arxiv.org/abs/1702.05970) | | arXiv | 2017 |
| FCN   | MRI | 肝，肝癌 | SurvivalNet: Predicting patient survival from diffusion weighted magnetic resonance images using cascaded fully convolutional and 3D convolutional neural networks [[pdf]](https://arxiv.org/abs/1702.05941) | | ISBI | 2017 |
| 3D-CNN | Diffusion MRI | 大脑 | q-Space Deep Learning: Twelve-Fold Shorter and Model-Free Diffusion MRI [[pdf]](http://ieeexplore.ieee.org/document/7448418/) (Section II.B.2) | | IEEE-TMI | 2016 |
| GAN   | MG | 心肌 | Adversarial Deep Structured Nets for Mass Segmentation from Mammograms [[pdf]](https://arxiv.org/abs/1710.09288) | [INbreast](http://medicalresearch.inescporto.pt/breastresearch/index.php/Get_INbreast_Database), [DDSM-BCRP](http://marathon.csee.usf.edu/Mammography/DDSM/BCRP/) | ISBI | 2018 |
| 3D-CNN | CT | 肝脏 | 3D Deeply Supervised Network for Automatic Liver Segmentation from CT Volumes [pdf](https://link.springer.com/content/pdf/10.1007%2F978-3-319-46723-8_18.pdf) | | MICCAI | 2017 |
| 3D-CNN | MRI | 大脑 | Unsupervised domain adaptation in brain lesion segmentation with adversarial networks [pdf](https://arxiv.org/pdf/1612.08894.pdf ) | | IPMI | 2017
| FCN | FUNDUS | 眼睛/视网膜 | A Fully Convolutional Neural Network based Structured Prediction Approach Towards the Retinal Vessel Segmentation [pdf](https://arxiv.org/pdf/1611.02064) | | ISBI | 2017

#### 正则化
| 计算机技术 | 医学技术 | 目标区域 | 标题| 数据库 | J/C | 年份 |
| ------ | ----------- | ----------- | ----------- |---|----------- | ---- |
| 3D-CNN | CT | 脊柱 | An Artificial Agent for Robust Image Registration [[pdf]](https://arxiv.org/pdf/1611.10336.pdf) | | | 2016 |

#### 回归

| 计算机技术 | 医学技术 | 目标区域 | 标题| 数据库 | J/C | 年份 |
| ------ | ----------- | ----------- | ----------- |---|----------- | ---- |
| 2.5D-CNN   | MRI | | Automated anatomical landmark detection ondistal femur surface using convolutional neural network [[pdf]](http://webpages.uncc.edu/~szhang16/paper/ISBI15_knee.pdf) | [OAI](https://oai.epi-ucsf.org/datarelease/)| ISBI | 2015|
| 3D-CNN | Diffusion MRI | 大脑 | q-Space Deep Learning: Twelve-Fold Shorter and Model-Free Diffusion MRI [[pdf]](http://ieeexplore.ieee.org/document/7448418/) (Section II.B.1) | [[HCP]](http://www.humanconnectome.org/) and other | IEEE-TMI | 2016 |

#### 图像重建
| 计算机技术 | 医学技术 | 目标区域 | 标题| 数据库 | J/C | 年份 |
| ------ | ----------- | ----------- | ----------- |---|----------- | ---- |
| CNN | CS-MRI | | A Deep Cascade of Convolutional Neural Networks for Dynamic MR Image Reconstruction [pdf](http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8067520) | | IEEE-TMI | 2017 |
| GAN | CS-MRI | | Deep Generative Adversarial Networks for Compressed Sensing Automates MRI [pdf](https://www.doc.ic.ac.uk/~bglocker/public/mednips2017/med-nips_2017_paper_7.pdf) | | NIPS | 2017 |

#### 图像合成
| 计算机技术 | 医学技术 | 目标区域 | 标题| 数据库 | J/C | 年份 |
| ------ | ----------- | ----------- | ----------- |---|----------- | ---- |
| GAN | RGB (Microscopy) | 红细胞 | Red blood cell image generation for data augmentation using Conditional Generative Adversarial Networks [[pdf]](https://arxiv.org/abs/1901.06219) | | arXiv | 2019 |
| GAN | MRI | 大脑 | Learning Data Augmentation for Brain Tumor Segmentation with Coarse-to-Fine Generative Adversarial Networks [[pdf]](https://arxiv.org/pdf/1805.11291.pdf) | | arXiv | 2018 |
| GAN | MRI | 大脑 | Medical Image Synthesis for Data Augmentation and Anonymization using Generative Adversarial Networks [[pdf]](https://arxiv.org/pdf/1807.10225.pdf) | | arXiv | 2018 |
| GAN | CT, MRI | 大脑 | GAN Augmentation: Augmenting Training Data using Generative Adversarial Networks [[pdf]](https://arxiv.org/pdf/1810.10863.pdf) | | arXiv | 2018 |
| GAN | CT | 肝脏 | GAN-based Synthetic Medical Image Augmentation for increased CNN Performance in Liver Lesion Classification [[pdf]](https://arxiv.org/pdf/1803.01229.pdf) | | arXiv | 2018 |

#### 其他

## 引用 
