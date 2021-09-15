# [A Simple Unified Framework for Detecting Out-of-Distribution Samples and Adversarial Attacks](https://arxiv.org/pdf/1807.03888.pdf)


## 들어가기에 앞서
- 수식을 보기 위해 
  - 크롬 앱스토어에서 MathJax Plugin for Github 설치 
  혹은
  - clone 한 뒤 vscode 에 markdown previewer 설치하고 보기  
- 내용 9장, refer 2장, Supplementary Material 9장(!!!)
- 이 논문이 처음 접할때 어렵다면 
  - 수학적인 내용 보다는 수리통계적 지식이 필요 (저도 모르는 내용들도 있음 - GDA, LDA 등)
  - 요구되는 개념들
    - Vector and Matrice calculation
    - Gaussian Dist. (사실 Normal Dist. ) 
    - Multivariate Distribution
    - Discriminat vs Generative (머피 책.)  
    - 사전 분포 vs 사후 분포  그리고 베이즈 정리 
    - 기타 등등

- 원문 발췌 및 아래 에 해석

- Links.. 부록에도 있음 
  - [Empirical distribution function](https://en.wikipedia.org/wiki/Empirical_distribution_function)
  - [Multivariate normal distribution](https://en.wikipedia.org/wiki/Multivariate_normal_distribution)
  - [Normal distribution](https://en.wikipedia.org/wiki/Normal_distribution)
  - [Mahalanobis distnace](https://en.wikipedia.org/wiki/Mahalanobis_distance)
  - [Gaussian Discriminant Analysis 1](https://www.eecs189.org/static/notes/n18.pdf)
  - [Gaussian Discriminant Analysis 2](https://msmskim.tistory.com/35)
  - [Gaussian Discriminant Analysis 3](https://gaussian37.github.io/ml-concept-gaussian_discriminant/)
  - [Linear discriminant analysis](https://en.wikipedia.org/wiki/Linear_discriminant_analysis)
  - [Mahalanobis distnace](https://en.wikipedia.org/wiki/Mahalanobis_distance)
    - 잡썰 1.: 수학을 공부한 입장에서는 위 definition 의 well-definedness 부터 따지게 된다. 수학은 계산만 하는 학문이 아니에요 ㅜㅜ
    - 잡썰 2.: KL Divergence 는 metric(distance)이 아닌 **차이** 다.  metric 이 되기위한 조건인 symmetric을 만족하지 못함.
      - [Metric (mathematics)](https://en.wikipedia.org/wiki/Metric_(mathematics))
      - [Kullback–Leibler divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence)
  - [The Matrix Cookbook](https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf)
  - [Matrix Calculus](https://ccrma.stanford.edu/~dattorro/matrixcalc.pdf)
  - [최대우도법](https://angeloyeo.github.io/2020/07/17/MLE.html)

## 핵심
1. softmax 로 사전 학습 
2. softmax layer 만 떼고, 직전 값( x -> **f(x)**) 들이 각 label 별로 Gaussian Distribution을 따름 (Under LDA assumption)
3. class 별 **f(x)** 의 mean, covariance 을 empirical 하게 추정
    - [Empirical distribution function](https://en.wikipedia.org/wiki/Empirical_distribution_function)
4. test sample **x** 에 대하여, confidence score **M(x)** 를 정의 :  Mahalanobis distance 가 제일 작은 C -> M(x) = (Mahalanobis distance)^2



## Abstrct

- 훈련된 분포로부터 충분히 멀리 떨어진 데이터를 detecting 하는 것은 classifier를 deploying 하기 위한 기본적인 요구사항, 그러나 softmax classifier 는 그러한 adnormal data 에 대해  highly overconfident posterior distributions 를 만드는 것으로 알려져 있다.

- pre-train 된 모든 softmax neural classifier 에 적용 가능한 simple method 제안
	- Gaussian discriminant analysis 로, class conditional Gaussian distribution w.r.t features.를 구함
	  - Gaussian distribution = Normal Distribution

- 이전에 제안된 방법들은 OOD나 adversarial 탐지 둘 중 하나에 대해  evaluate 되었지만 제안된 방법은 두 경우 모두에 대해 SOTA

- 제안한 방법이, data 에 노이즈가 많거나, 적은 경우에도 더 robust 하다는 것을 발견

- 또한 class-incremental learning 에 적용 할 수 있다: OOD sample 이 발견되면 추가 훈련 없이 새로운 class를 잘 구별.


## 1. Introduction


 The predictive uncertainty of DNNs is closely related to the problem of detecting abnormal samples that are drawn far away from in-distribution (i.e.,  distribution of training samples) statistically or adversarially.

- predictive uncertainty of DNN 은 훈련 샘플의 분포로부터 멀리 떨어진 abnormal sample detecting 에 밀접히 연관되어 있음.



이전 연구들 	
- For detecting OOD -> 13, 21  
  - Hendrycks, Dan and Gimpel, Kevin. A baseline for detecting misclassified and out-ofdistribution examples in neural networks. In ICLR, 2017.[13](https://arxiv.org/pdf/1610.02136.pdf)
  - Liang, Shiyu, Li, Yixuan, and Srikant, R. Principled detection of out-of-distribution examples in neural networks. In ICLR, 2018. [21](https://www.researchgate.net/publication/317418851_Principled_Detection_of_Out-of-Distribution_Examples_in_Neural_Networks)
- For detecting adversarial sample -> 7, 22
  - Feinman, Reuben, Curtin, Ryan R, Shintre, Saurabh, and Gardner, Andrew B. Detecting adversarial samples from artifacts. arXiv preprint arXiv:1703.00410, 2017. [7](https://arxiv.org/pdf/1703.00410.pdf)
  - Ma, Xingjun, Li, Bo, Wang, Yisen, Erfani, Sarah M, Wijewickrema, Sudanthi, Houle, Michael E, Schoenebeck, Grant, Song, Dawn, and Bailey, James. Characterizing adversarial subspaces using local intrinsic dimensionality. In ICLR, 2018.[22](https://arxiv.org/pdf/1801.02613.pdf)


### Contribution 

In this paper, we propose a simple yet effective method, which is applicable to any pre-trained softmax neural classifier (without re-training) for detecting abnormal test samples including OOD and adversarial ones.

Our high-level idea is to measure the probability density of test sample on feature spaces of DNNs utilizing the concept of a “generative” (distance-based) classifier.

- **generative classifier를 활용하여 test sample 의 확률 밀도를 measure.**

Specifically, we assume that pre-trained features can be fitted well by a class-conditional Gaussian distribution since its posterior distribution can be shown to be equivalent to the softmax classifier under Gaussian discriminant analysis (see Section 2.1 for our justification)

- 

Under this assumption, we define the confidence score using the Mahalanobis distance with respect to the closest classconditional distribution, where its parameters are chosen as empirical class means and tied empirical covariance of training samples

- **이 가정 하에서,  the Mahalanobis distance with respect to the closest classconditional distribution을 활용하여 confidence score를 정의** 
  - 각 class가 개별적인 분포를 따름 -> 각 클래스별로 Mahalanobis distance 계산 가능 -> Mahalanobis distance 가 제일 적을 쪽으로 confidence score 를 정의

To the contrary of conventional beliefs, we found that using the corresponding generative classifier does not sacrifice the softmax classification accuracy. 

- **기존에 믿어진 것과 반대로, corresponding generative classifier를 사용하는 것이 softmax 정확도를 희생시키지 않음**

Perhaps surprisingly, its confidence score outperforms softmax-based ones very strongly across multiple other tasks: detecting OOD samples, detecting adversarial samples and class-incremental learning


## 2. Mahalanobis distance-based score from generative classifier


### 2.1 Why Mahalanobis distance-based score?

#### Derivation of generative classifiers from softmax ones.

- Notation 
  - input  
    $$ \mathbf{x} \in \chi  $$ 
  - label
    $$ y \in \mathsf{Y}  = \{ 1, 2, \cdots, \mathbf{C} \} $$ 
  - softmax classifier
    $$ P( y = c | \mathbf{x} )  =  \frac{\exp({\mathbf{w}_{c}^{T} f(\mathbf{x}) + b_{c} )}}{ \sum_{c'} \exp({\mathbf{w}_{c'}^{T} f(\mathbf{x}) + b_{c'} )}   }$$ 
    where 
    $$f( \cdot   )$$
    denots the output of the penultimate layer of DNNs. (penultimate layer 는 softmax 바로 이전의 layer)


Then, without any modification on the pre-trained softmax neural classifier, we obtain a generative classifier assuming that a class-conditional distribution follows the multivariate Gaussian distribution.  

Specifically, we define C class-conditional Gaussian distributions with a tied covariance Σ: P (f(x)|y = c) = N (f(x)|µc, Σ), where µc is the mean of multivariate Gaussian distribution of class c ∈ {1, ..., C}. 

- 특히 C class-conditional Gaussian distributions 을 정의 할 수 있음
  -  covariance 
    $$ \Sigma: P (f(x)|y = c) = N (f(x)|\mu_{c}, \Sigma)$$
  - 의미: softmax 직전 layer 의 값(vecotr)들이, 각 class 별로 , 정규분포를 따른다.

Here, **our approach is based on a simple theoretical connection between GDA and the softmax classifier: the posterior distribution defined by the generative classifier under GDA with tied covariance assumption is equivalent to the softmax classifier** (see the supplementary material for more details). 

- our approch 는 **GDA와 softmax classifier 사이의 간단한 이론에 기반**하고 있다.

Therefore, the pre-trained features of the softmax neural classifier f(x) might also follow the class-conditional Gaussian distribution.

- 따라서 **pre-trained features of the softmax neural classifier f(x) 또한 class-conditional Gaussian distribution** 을 따른다.


To estimate the parameters of the generative classifier from the pre-trained softmax neural classifier, we compute the empirical class mean and covariance of training samples {(x1, y1), . . . ,(xN , yN )}:

- 사전 훈련된 softmax neural classifier로부터,  generative classifier의 prameter(통계에서 parameter 라 함은 mean 과 (co)variance 의미) 추정을 위해 다음과 같이 empirical class mean and covariance 를 계산한다. 
- $$  \hat{\mu_{c}} = \frac{1}{N_{c}}\Sigma_{i: y_{i} = c}f(x_i)  $$
- $$  \hat{\Sigma} = \frac{1}{N}\Sigma_{c} \Sigma_{i:y_{i} = c}  (f(x_{i}) - \hat{\mu_{c}} ) (f(x_{i}) - \hat{\mu_{c}} ) ^{T} $$

where N_c is the number of training samples with label c.

- 위 계산 이해를 위해 다음을 참고: [Multivariate normal distribution](https://en.wikipedia.org/wiki/Multivariate_normal_distribution)

 This is equivalent to fitting the classconditional Gaussian distributions with a tied covariance to training samples under the maximum likelihood estimator.

#### Mahalanobis distance-based confidence score


- Def. [Mahalanobis distnace](https://en.wikipedia.org/wiki/Mahalanobis_distance)
  - observation 과 distribution 간의 거리.
  - By similar sense, we can also define distance between two random variable.

Define the confidence score **M(x)**

$$ M(x) = \max_{c} \{ - (f(x) - \hat{\mu_{c}}) ^{T}  \hat{\Sigma}^{-1}  (f(x) - \hat{\mu_{c}})   \} = \min_{c} \{ (f(x) - \hat{\mu_{c}}) ^{T}  \hat{\Sigma}^{-1}  (f(x) - \hat{\mu_{c}})  \} $$


#### Experimental supports for generative classifiers.

To evaluate our hypothesis that trained features of DNNs support the assumption of GDA, we measure the classification accuracy as follows:

$$\hat{y}(x) = \arg \min_{c} \{ (f(x) - \hat{\mu}_{c})^{T} \hat{\Sigma}^{-1} (f(x) - \hat{\mu}_{c}) \}$$ 

We remark that this corresponds to predicting a class label using the posterior distribution from generative classifier with the uniform class prior. 

- 즉, class label 을 posterior distrubution from generative classifier with the unisform class prior 로 예측.

Interestingly, we found that the softmax accuracy (red bar) is also achieved by the Mahalanobis distance-based classifier (blue bar), while conventional knowledge is that a generative classifier trained from scratch typically performs much worse than a discriminative classifier such as softmax. 

- 관습접으로 알려진 것(generative classifier trained from scratch typicallly performs much worse than a discriminative classifier such as softmax) 과 다르게, the Mahalanobis distance-based classifier 로도 softmax accuracy 달성 : Figure 1 - (b)






### 2.2 Calibration techniques
#### Input pre-processing

**To make in- and out-of-distribution samples more separable**, we consider adding a small controlled noise to a test sample. 
- 인도메인 과 OOD 샘플을 좀 더 떼어놓게 하기 위해, test sample 에 작은 nodise 를 더하는 것을 고려.
 
Specifically, for each test sample x, we calculate the pre-processed sample xb by adding the small perturbations as follows:
- 다음과 같이 small pertubations 을 더해서 pre-processed sample 을 계산

- $$ \hat{x} = x + \epsilon \, sign(\bigtriangledown_{x} \; (f(x) - \hat{mu}_{\hat{c}})^{T} \hat{\Sigma}^{-1}  \; (f(x) - \hat{mu}_{\hat{c}})  )$$
  - 위 식은 의의는 아래 문서 참고.
  - ε is a magnitude of noise and bc is the index of the closest class.

We remark that the noise is generated to increase the proposed confidence score (2) unlike adversarial attacks [10](https://arxiv.org/pdf/1412.6572.pdf).

- 악의적 공격과 달리, 제안된 confidence score 를 높이기 위해 노이즈가 사용됨.

In our experiments, such perturbation can have stronger effect on separating the in- and out-of-distribution samples.
- 실험에서 이런 미동이 인도메인/아운돔인 의 분리에 강한 영향을 가질 수 있었다.


We remark that similar input pre-processing was studied in [21], where the perturbations are added to increase the softmax score of the predicted label.
- 21 에서 비슷한 input pre-processing 이 연구 되었다 ( predicted label 의 softmax 를 increse 하기 위해)

However, our method is different in that the noise is generated to increase the proposed metric.
- 그러나 우리의 방법은 noise 가 제안된 metric 을 증가하기 위해 만들어 진다는 것에서 다르다.


#### Feature ensemble

further improve the performance, we consider measuring and combining the confidence scores from not only the final features but also the other low-level features in DNNs.

- 성능 향상을 위해, final features 에서 confidonce socore 계산하는게 아니라, dnn 의 low-level features 도 측정하고 조합한다.

Formally, given training data, we extract the l-th hidden features of DNNs, denoted by f_l(x), and compute their empirical class means and tied covariances, i.e., µ^_l,c and Σ^_l. 

- f_l(x) 를 l-th hidden layer 의 feature라고 하자. 그리고  empirical class means and tied covariances 을 계산하고 µ^_l,c and Σ^_l. 라 하자.

Then, for each test sample x, we measure the confidence score from the l-th layer using the formula in (2).

- 그러면 각 test sample x 에 대해서 l th layer 에서 confidence score 를 측정 할 수 있다.

One can expect that this simple but natural scheme can bring an extra gain in obtaining a better calibrated
score by extracting more input-specific information from the low-level features. 

- 간단하지만 자연스러운 이 스킴은, low-level features 로 부터 input-specific 한 정보를 뽑아 calibrated
score 를 얻 을 수 있는 추가적인 이득을 가져다 줄 수 있다.


We measure the area under ROC (AUROC) curves of the threshold-based detector using the confidence score in (2) computed at different basic blocks of DenseNet [14] trained on CIFAR-10 dataset, where the overall trends on ResNet are similar.

- CIFAT-10 에서 훈련된 DenseNet 에 제안된 confidence score 를 사용한 detector 의 ROC 커브 측정. ResNet 도 전반적으로 유사.


Figure 2 shows the performance on various OOD samples such as SVHN [28], LSUN [32], TinyImageNet and adversarial samples generated by DeepFool [26], where the dimensions of the intermediate features are reduced using average pooling (see Section
3 for more details). 

As shown in Figure 2, the confidence scores computed at low-level features often provide better calibrated ones compared to final features (e.g., LSUN, TinyImageNet and DeepFool). 

- 그림 2 에서 볼 수 있듯이 low-level 피처 에서의 confidence 는 final 피처 보다 좀더 나은 calbrated score 제공.

To further improve the performance, we design a feature ensemble method as described in Algorithm 1. 

- 성능을 더 향상 시키기 위해, feature ensemble method 를 Algorithm 1 에 디자인 했다.

We first extract the confidence scores from all layers, and then integrate them by weighted averaging: Σ α_l·M_l(x), where M_l(·) and α_ is the confidence score at the l-th layer and its weight, respectively. 

- 우리는 먼저 모든 layer 에서 confidence score 를 계산하고 weighted averaging 한다.

**In our experiments, following similar strategies in [22], we choose the weight of each layer α_l by training a logistic regression detector using validation samples.**

- α_l 은 validation sample 을 사용하여  logistic regression detector 를 훈련하여 선택.

We remark that such weighted averaging of confidence scores can prevent the degradation on the overall performance even in the case when the confidence scores from some layers are not effective:  the trained weights (using validation) would be nearly zero for those ineffective layers.

- weighted averaging of confidence scores 는 어떤 lyaer 의 confidence 가 효과적이지 않은 경우에도 전반적인 성능이 저하되는 것을 막아준다. : validation 을 사용하여 훈련된 가중치는  ineffective layers 에 대해 거의 0 이 된다.


### 2.3 Class-incremental learning using Mahalanobis distance-based score


As a natural extension, we also show that the Mahalanobis distance-based confidence score can be utilized in class-incremental learning tasks [29]: a classifier pre-trained on base classes is progressively updated whenever a new class with corresponding samples occurs.

- Mahalanobis distance-based confidence score 가 class-incremental learning tasks 에 적용될수 있다. : 사전 훈련된 classifier 가 새로운 class에 대응되는 샘플이 발생하면 점진적으로 업데이트 됨.

This task is known to be challenging since one has to deal with catastrophic forgetting [24] with a limited memory. 
- 이 task는 제한된 메모리으로 치명적인 forgetting  을 처리해야 해서 어려운 것으로 알려짐.

To this end, recent works have been toward developing new training methods which involve a generative model or data sampling, but adopting such training methods might incur expensive back-and-forth costs. 
- 최근 연구에서는 a generative model 나 data sampling과 연관된 새로운 훈련 방법을 개발하는 방향으로 진행. 그러나 이러한 방법은 비싼 back-and-forth costs 발생.

Based on the proposed confidence score, we develop a simple classification method without the usage of complicated training methods. 
- 제안된 신뢰 점수를 기반으로 복잡한 traainig 의 사용 없이 간단한 분류 방법을 개발.

To do this, we first assume that the classifier is well pre-trained with a certain amount of base classes, where the assumption is quite reasonable in many practical scenarios.1

- 이를 위해 먼저 분류기가 특정 양의 기본 클래스로 사전 훈련되어있다고 가정한다. 이 가정은 실제 시나리오에서 아주 합리적이다.

In this case, one can expect that not only the classifier can detect OOD samples well, but also might be good for discriminating new classes, as the representation learned with the base classes can characterize new ones. 
- 이 경우 classifier OOD 샘플을 잘 감지할 뿐만 아니라, 새로운 클래스를 구별하는데에도 좋다고 예상 할 수 있다, 

Motivated by this, we present a Mahalanobis distance-based classifier based on (3), which tries to accommodate a new class by simply computing and updating the class mean and covariance, as described in Algorithm 2. 
- 이 방법에 motivated 되어,  Algorithm 2 에 설명된 것 처럼 class 평균과 공분산을 계산하고 업데이트하여 새로운 클래스를 수용하려고 함.


The class-incremental adaptation of our confidence score shows its potential to be applied to a wide range of new applications in the future.
- confidence score  의 class-incremental 적용은 앞으로  넓은 범위의 새로운 응용이 가능성을 보여준다.



## 3.  Experimental Results

TBA


## 4. Conclusion

In this paper, we propose a simple yet effective method for detecting abnormal test samples including both out-of-distribution and adversarial ones. 

- detecting abnormal test samples 을 위한 효과적인 방법 제안

In essence, our main idea is inducing a generative classifier under LDA assumption, and defining new confidence score based on it. 

- Main Idea :  Generative classifier under LDA assumption and define confidence score.

With calibration techniques such as input pre-processing and feature ensemble, our method performs very strongly across multiple tasks: detecting out-of-distribution samples, detecting adversarial attacks and classincremental learning. 

- 캘리브레이션 테크닉과 feature ensemble 을 통해 OOD, Adversarial attack 그리고 classincremental learning 3 가지 task에  perform.

We also found that our proposed method is more robust in the choice of its hyperparameters as well as against extreme scenarios, e.g., when the training dataset has some noisy, random labels or a small number of data samples. 

- extreme scenario 뿐만 아니라 하이퍼 파리미터 선택에도 더 강건.

We believe that our approach have a potential to apply to many other related machine learning tasks, e.g., active learning [8], ensemble learning [19] and few-shot learning [31].

- 본 논문의 접근이 다른 task 에도 적용할 수 있다고 봄.





## Supplementary Material: 

- Links.. 부록에도 있음 
  - [Empirical distribution function](https://en.wikipedia.org/wiki/Empirical_distribution_function)
  - [Multivariate normal distribution](https://en.wikipedia.org/wiki/Multivariate_normal_distribution)
  - [Normal distribution](https://en.wikipedia.org/wiki/Normal_distribution)
  - [Mahalanobis distnace](https://en.wikipedia.org/wiki/Mahalanobis_distance)
  - [Gaussian Discriminant Analysis 1](https://www.eecs189.org/static/notes/n18.pdf)
  - [Gaussian Discriminant Analysis 2](https://msmskim.tistory.com/35)
  - [Gaussian Discriminant Analysis 3](https://gaussian37.github.io/ml-concept-gaussian_discriminant/)
  - [Linear discriminant analysis](https://en.wikipedia.org/wiki/Linear_discriminant_analysis)
  - [Mahalanobis distnace](https://en.wikipedia.org/wiki/Mahalanobis_distance)
    - 잡썰 1.: 수학을 공부한 입장에서는 위 definition 의 well-definedness 부터 따지게 된다. 수학은 계산만 하는 학문이 아니에요 ㅜㅜ
    - 잡썰 2.: KL Divergence 는 metric(distance)이 아닌 **차이** 다.  metric 이 되기위한 조건인 symmetric을 만족하지 못함.
      - [Metric (mathematics)](https://en.wikipedia.org/wiki/Metric_(mathematics))
      - [Kullback–Leibler divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence)
  - [The Matrix Cookbook](https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf)
  - [Matrix Calculus](https://ccrma.stanford.edu/~dattorro/matrixcalc.pdf)
  - [최대우도법](https://angeloyeo.github.io/2020/07/17/MLE.html)

### A Preliminaries for Gaussian discriminant analysis

In this section, we describe the basic concept of the discriminative and generative classifier [27].
  - Refer #27 : Murphy, Kevin P. Machine learning: a probabilistic perspective. 2012. [책 다운](http://noiselab.ucsd.edu/ECE228/Murphy_Machine_Learning.pdf)
    - 위 ebook 에서 4.2 Gaussian discriminant analysis p101
    - 위 ebook 에서 8.6 Generative vs discriminative classifiers p267
        - A further difference between these models is the way they are trained. When fitting a discriminative model, we usually maximize the conditional log likelihood )N i=1 log p(yi|xi, θ), whereas when fitting a generative model, we usually maximize the joint log likelihood, )N i=1 log p(yi, xi|θ). It is clear that these can, in general, give different results (see Exercise 4.20).
        - generative 와 discriminative 의 차이는 그것들이 학습되는 방식에 있다. discriminative model 에 fitting 될때는 우리는 대개 conditional log liklihood 를 maximize 한다. 반면에 generative model 에 fiiting  할때는 joint log liklihood 를 maximize 한다. 
          - conditional log likelihood 
          $$\Sigma_{i=1}^{N} \log{p(y_i | x_i , \theta )}$$
          - joint log likelihood
          $$\Sigma_{i=1}^{N} \log{p(y_i,  x_i | \theta )}$$

    
    - When the Gaussian assumptions made by GDA are correct, the model will need less training data than logistic regression to achieve a  certain level of performance, but if the Gaussian assumptions are incorrect, logistic regression will do better (Ng and Jordan 2002). This is because discriminative models do not need to model the distribution of the features. This is illustrated in Figure 8.10.


Formally, denote the random variable of the input and label as x ∈ X and y ∈ Y = {1, · · · , C}, respectively. 

For the classification task, the discriminative classifier directly defines a posterior distribution P(y|x), i.e., learning a direct mapping between input x and label y.

A popular model for discriminative classifier is softmax classifier which defines the posterior distribution as follows:

$$ P( y = c | \mathbf{x} )  =  \frac{\exp({\mathbf{w}_{c}^{T} f(\mathbf{x}) + b_{c} )}}{ \sum_{c'} \exp({\mathbf{w}_{c'}^{T} f(\mathbf{x}) + b_{c'} )}   }$$ 

, where
 $$w_{c}$$
 and 
 $$b_{c}$$
are weights and bias for a class c, respectively.

- softmax 는 위와 같이 정의


In contrast to the discriminative classifier, the generative classifier defines the class conditional distribution P (x|y) and class prior P (y) in order to indirectly define the posterior distribution by specifying the joint distribution P (x, y) = P (y) P (x|y).

- discriminative 와 대조적으로 generative 는 prior 와 condtional dist. 을 통해 posterior distribution 을 inderct 하게 정의
  - P (x, y) = P (y) P (x|y)

Gaussian discriminant analysis (GDA) is a popular method to define the generative classifier by assuming that the class conditional distribution follows the multivariate Gaussian distribution and the class prior follows Bernoulli distribution: 

$$ P(x|y=c) = N(x | \mu_{c}, \Sigma_{c}),  P(y=c) = \frac{\beta_{c}}{\Sigma_{c'} \beta_{c'}}$$
, where µc and Σc are the mean and covariance of multivariate Gaussian distribution, and βc is the unnormalized prior for class c. 

- class prior ~ 베르누이 분포,  class conditional distribution  ~ multivariate Gaussian distribution  -> GDA
  - GDA definition 을 찾아봐도 the class prior follows Bernoulli distribution 는 언급이 없음.  

This classifier has been studied in various machine learning areas (e.g., semi-supervised learning [17] and incremental learning [29]).

- GDA 는 많은 머신러닝 분야에서 연구됨(semi-supervised and incremental learning) 

In this paper, we focus on the special case of GDA, also known as the linear discriminant analysis (LDA). 
- 이 논문에선, GDA의 special case 인 LDA에 founcs on. 

In addition to Gaussian assumption, LDA further assumes that all classes share the same covariance matrix, i.e., Σc = Σ.
- LDA 는 GDA 가정 + 각 class 분포의 covariance matrix 가 같다.

Since the quadratic term is canceled out with this assumption, the posterior distribution of generative classifier can be represented as follows: 

$$ P( y = c | x) =  \frac{ P( y = c ) P( x|y = c )  }{P(x)}= \frac{P(y=c) P(x|y=c) }{ \Sigma_{c'}{P(y=c')}  P(x|y=c') }$$

- 첫번째 등식 : 베이즈 정리, 두번째 등식 : sum over all classes
- 베이즈 정리: 
  - $$P(A | B) = \frac{P(B|A)P(A)}{P(B)} $$
- [베이즈 정리](https://ko.wikipedia.org/wiki/%EB%B2%A0%EC%9D%B4%EC%A6%88_%EC%A0%95%EB%A6%AC)

- GDA 가정에 의해, 
   $$ P(x|y=c) = N(x | \mu_{c}, \Sigma_{c}) = \frac{1}{ (2\pi)^{\frac{n}{2} }\Sigma_{c}^{\frac{1}{2}}} \exp (- \frac{1}{2}(x-\mu_{c})^{T} \Sigma_{c}^{-1}(x-\mu_{c}) ) $$ 
   여기서 n 은 random variable 의 차원 

- 또한 prior 는 
  $$P(y=c) = \frac{\beta_{c}}{\Sigma_{c'} \beta_{c'}}=\frac{ \exp \log \beta_{c}}{\Sigma_{c'} \beta_{c'}} =  \frac{ \exp \log \beta_{c}}{N}$$

- LDA 가정에 의해, 
   $$ \Sigma_{c'} = \Sigma \quad \forall  c' \in C  $$
   
   따라서  P(x|y=c) 에 나타나는 quadratic form 은 상수. 따라서 약분 가능


- $$ \frac{P(y=c) P(x|y=c) }{ \Sigma_{c'}{P(y=c')}  P(x|y=c') }$$
 을 다시 쓰면, 
  -  분자 
  $$  \frac{ \exp \log \beta_{c}}{N}  \frac{1}{ (2\pi)^{\frac{n}{2} }\Sigma^{\frac{1}{2}}} \exp (- \frac{1}{2}(x-\mu_{c})^{T} \Sigma^{-1}(x-\mu_{c})  $$
  -  분모
  $$\Sigma_{c' \in C }  \frac{ \exp \log \beta_{c'}}{N}  \frac{1}{ (2\pi)^{\frac{n}{2} }\Sigma^{\frac{1}{2}}} \exp (- \frac{1}{2}(x-\mu_{c'})^{T} \Sigma^{-1}(x-\mu_{c'}))  $$ 

- 따라서 
$$P( y = c | x) =  \frac{ \exp \log \beta_{c}  \exp (- \frac{1}{2}(x-\mu_{c})^{T} \Sigma^{-1}(x-\mu_{c}) ) }{ \Sigma_{c'} \exp \log \beta_{c'}  \exp (- \frac{1}{2}(x-\mu_{c'})^{T} \Sigma^{-1}(x-\mu_{c'})) }$$
로 정리 됨.

- $$ - \frac{1}{2}(x-\mu_{c})^{T} \Sigma^{-1}(x-\mu_{c})  $$
   을 정리하면 

  $$ -\frac{1}{2} x^{T} \Sigma^{-1} x  + \frac{1}{2} \mu_{c}^{T} \Sigma^{-1} x  +   \frac{1}{2} x^{T} \Sigma^{-1} \mu_{c} - \frac{1}{2}  \mu_{c}^{T} \Sigma^{-1} \mu_{c}  $$

  x 는 c와 무관하므로 상수, 따라서
  
  $$ -\frac{1}{2} x^{T} \Sigma^{-1} x $$

  는 약분 가능

  
- 따라서  

$$ P( y = c | x) =  \frac{ \exp \log \beta_{c}  \exp (- \frac{1}{2}(x-\mu_{c})^{T} \Sigma^{-1}(x-\mu_{c}) ) }{ \Sigma_{c'} \exp \log \beta_{c'}  \exp (- \frac{1}{2}(x-\mu_{c'})^{T} \Sigma^{-1}(x-\mu_{c'})) }$$

는 

$$P( y = c | x) =  \frac{ \exp  ( \log \beta_{c} +  \mu_{c}^{T} \Sigma^{-1} x - \frac{1}{2} \mu_{c}^{T} \Sigma^{-1}\mu_{c}) }{ \Sigma_{c'} \exp  ( \log \beta_{c'} +  \mu_{c'}^{T} \Sigma^{-1} x - \frac{1}{2} \mu_{c'}^{T} \Sigma^{-1}\mu_{c'}) } = \frac{ \exp  (   \mu_{c}^{T} \Sigma^{-1} x - \frac{1}{2} \mu_{c}^{T} \Sigma^{-1}\mu_{c} + \log \beta_{c} ) }{ \Sigma_{c'} \exp  ( \mu_{c'}^{T} \Sigma^{-1} x - \frac{1}{2} \mu_{c'}^{T} \Sigma^{-1}\mu_{c'} + \log \beta_{c'} ) }$$


  
    


One can note that the above form of posterior distribution is equivalent to the softmax classifier by considering 

$$µ_{c}^{T} \Sigma^{−1}  $$

and 
$$-\frac{1}{2}µ_{c}^{T} \Sigma^{−1}µ_{c} + \log \beta_{c}  $$
as weight and bias of it, respectively. 


**This implies that x might be fitted in Gaussian distribution during training a softmax classifier**


### B Experimental setup

### C More experimental results

### D Evaluation on ImageNet dataset
