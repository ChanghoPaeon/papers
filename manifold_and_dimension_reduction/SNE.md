* * *


## [Stochastic Neighbor Embedding](https://cs.nyu.edu/~roweis/papers/sne_final.pdf)


- 원리

  원본 데이터 x_i 들에 대해,  x_i, x_j 의 유사도를 정의.
  차원 축소된 y_i에 대해, y_i, y_j 의 유사도를 정의.
  두 공간상 유사도 들의 차이가 적어지도록 학습

  - High Dimension 에서 x_i 와 x_j 의 유사도 :  p_ij = 
    - simgma : that make the entpory of distribution over neighbors equal to log k. Here, k is the effective number of local neighbors or “perplexity” and is chosen by hand.
  
  - Low Dimension 에서  x_i 와 x_j 의 유사도 : q_ij = x_i의 low-dimension image y_i 에 대해 p_ij 와 유사하게 정의 (simga 나누는거 빠짐.
  
  - Cost: KL(P|Q) 로 정의 -> P 와 Q가 유사할수돍 작아지고 P가 커질수록 가중

- 유사도 수식 설명 :
  - p_ij : x_i , x_j 가 가까우면 차가 줄고 -> 분자 커짐

- [T-SNE wiki](https://en.wikipedia.org/wiki/T-distributed_stochastic_neighbor_embedding)

- T-SNE: SNE 에서 축소된 공간에서의 분포를 Gaussian neighborhoods but with a fixed variance 대신 t-student 사용


- [CSC 411 Lecture 13:t-SNE](https://www.cs.toronto.edu/~jlucas/teaching/csc411/lectures/lec13_handout.pdf)

- [t-SNE 개념과 사용법](https://gaussian37.github.io/ml-concept-t_sne/)
  - 알고리즘의 시각적 이해를 돕는 그림이 있습니다.

- [차원축소 시각화 도구 t - SNE Stochastic Neighbor Embedding](https://m.blog.naver.com/xorrms78/222112752837)
  - 수식을 간략히 이해하기 좋음

- [t-Stochastic Neighbor Embedding (t-SNE) 와 perplexity](https://lovit.github.io/nlp/representation/2018/09/28/tsne/)
  - 수식의 의미를 깊게 이해하기 좋음
    - 목차 : How to embed high dimensional vectors to low dimensional space


- [Dimension Reduction PCA, tSNE, UMAP](https://www.bioinformatics.babraham.ac.uk/training/10XRNASeq/Dimension%20Reduction.pdf)

#### t-SNE
 - SNE 에서 가우시안 대신 t-student 분포

* * *

### Supplementary
#### links
 - [d-SNE](http://dmqm.korea.ac.kr/uploads/seminar/20200522_d-SNE_%EC%A0%95%EC%8A%B9%EC%84%AD%20final.pdf)

- [CSC 411 Lecture 13:t-SNE](https://www.cs.toronto.edu/~jlucas/teaching/csc411/lectures/lec13_handout.pdf)

-[Visualizing Data Using t-SNE](https://www.youtube.com/watch?v=RJVL80Gg3lA&list=UUtXKDgv1AVoG88PLl8nGXmw)


#### 정규 분포의 표본 평균과 모평균의 추정 
- 판서
  - (1,2,3,4) 에서 sample 2. -> X bar 설명.

##### [Student's t-distribution](https://en.wikipedia.org/wiki/Student%27s_t-distribution)

- navie 한 이해: 정규 분포를 따르는  확률 변수에서 표본 추출 -> 표본 평균 계산(확률 변수 <- 표본이 어떻게 뽑히느냐 따라 달라짐) -> 이를 표본의 분산으로 표준화 (확률변수).   이것들의 분포가 t-student distribution ,  

- 특징 : 정규분포 대비 tail 이 두꺼움


- 목적 : 모집단이 정규분포를 하더라도 분산 σ²이 알려져 있지 않고 표분의 수가 적은 경우에, 평균 μ에 대한 신뢰구간 추정 및 가설검정
  - 고등학교 때 배운, 모평균 추정은 모집단의 표준 편차를 알때, Z 를 활용하여 추정. 
   - 고등학교때 문제들은 항상 모집단의 표준편차가 주어졌었음.


- [t분포는 무엇이며, 언제 그리고 왜 쓸까?](https://m.blog.naver.com/definitice/221031927257)


- t 분포는  infinite mixed Gaussian distribution

#### Crowding Problem
- we don’t have enough room to accommodate all neighbors
  - In 2D a point can have a few neighbors at distance one all far from each other - what happens when we embed in 1D?
