* * *
### [UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction](https://arxiv.org/pdf/1802.03426.pdf)



#### Abstract

 UMAP (Uniform Manifold Approximation and Projection) is a novel manifold learning technique for dimension reduction. UMAP is constructed
from a theoretical framework based in Riemannian geometry and algebraic topology. e result is a practical scalable algorithm that is applicable to real world data. The UMAP algorithm is competitive with t-SNE for visualization quality, and arguably preserves more of the global structure with superior run time performance. Furthermore, UMAP has no computational restrictions on embedding dimension, making it viable as a general purpose dimension reduction technique for machine learning.

#### 1 Introduction

#### 2 Theoretical Foundations for UMAP

 The theoretical foundations for UMAP are largely based in manifold theory and topological data analysis. Much of the theory is most easily explained in the language of topology and category theory. Readers may consult [39], [49] and [40] for background. Readers more interested in practical computational aspects of the algorithm, and not necessarily the theoretical motivation for the computations involved, may wish to skip this section. 
 
 Readers more familiar with traditional machine learning may find the relationships between UMAP, t-SNE and Largeviz located in Appendix C enlightening. Unfortunately, this purely computational view fails to shed any light upon the reasoning that underlies the algorithmic decisions made in UMAP. Without strong theoretical foundations the only arguments which can be made about algorithms amount to empirical measures, for which there are no clear universal choices for unsupervised problems. 
 
 At a high level, UMAP uses local manifold approximations and patches together their local fuzzy simplicial set representations to construct a topological representation of the high dimensional data. Given some low dimensional representation of the data, a similar process can be used to construct an equivalent topological representation. UMAP then optimizes the layout of the data representation in the low dimensional space, to minimize the cross-entropy between the two topological representations
 
 The construction of fuzzy topological representations can be broken down into two problems: approximating a manifold on which the data is assumed to lie; and constructing a fuzzy simplicial set representation of the approximated manifold. In explaining the algorithm we will first discuss the method of approximating the manifold for the source data. Next we will discuss how to construct a fuzzy simplicial set structure from the manifold approximation. Finally, we will discuss the construction of the fuzzy simplicial set associated to a low dimensional representation (where the manifold is simply Rd ), and how to optimize the representation with respect to our objective function


#### 3 A Computational View of UMAP

1. There exists a manifold on which the data would be uniformly distributed
2. The underlying manifold of interest is locally connected
3. Preserving the topological structure of this manifold is the primary goal

As highlighted in Appendix C any algorithm that attempts to use a  mathematical structure akin to a k-neighbour graph to approximate a manifold must follow a similar basic structure.
• Graph Construction
1. Construct a weighted k-neighbour graph
2. Apply some transform on the edges to ambient local distance.
3. Deal with the inherent asymmetry of the k-neighbour graph.
• Graph Layout
1. De랴ne an objective function that preserves desired characteristics of this k-neighbour graph.
2. Find a low dimensional representation which optimizes this objective function.

##### 3.1 Graph Construction


The selection of ρi derives from the local-connectivity constraint described in Section 2.2. In particular it ensures that xi connects to at least one other data point with an edge of weight 1; this is equivalent to the resulting fuzzy simplicial set being locally connected at xi. In practical terms this significantly improves the representation on very high dimensional data where other algorithms such as t-SNE begin to suffer from the curse of dimensionality. The selection of σi corresponds to (a smoothed) normalisation factor, defining the Riemannian metric local to the point xi as described in Section2.1

##### 3.2 Graph Layout
 - The forces described above are derived from gradients optimising the edge-wise cross-entropy between the weighted graph G, and an equivalent weighted graph H constructed from the points {yi}i=1..N . That is, we are seeking to position points yi such that the weighted graph induced by those points most closely approximates the graph G, where we measure the difference between weighted graphs by the total cross entropy over all the edge existence probabilities. Since the weighted graph G captures the topology of the source data, the equivalent weighted graph H constructed from the points {yi}i=1..N matches the topology as closely as the optimization allows, and thus provides a good low dimensional representation of the overall topology of the data

 - 두 그래프 G와 H에 대해 엣지의 존재 확률 의 total cross entropy 를 difference로 measure 하여 학습.

#### 4 Implementation and Hyper-parameters

##### 4.3 Hyper-parameters

1. n, the number of neighbors to consider when approximating the local metric;
2. d, the target embedding dimension;
3. min-dist, the desired separation between close points in the embedding space; and
4. n-epochs, the number of training epochs to use when optimizing the low dimensional representation

#### 5 Practical Efficacy
##### 5.1 Qualitative Comparison of Multiple Algorithms
![image](https://user-images.githubusercontent.com/27984736/139658424-3efc8190-f0f5-4630-a56a-a95bc2cb0218.png)

##### 5.2 Quantitative Comparison of Multiple Algorithms
 
 We compare UMAP, t-SNE, LargeVis, Laplacian Eigenmaps and PCA embeddings with respect to the performance of a k-nearest neighbor classifier trained on the embedding space for a variety of datasets
 
 ![image](https://user-images.githubusercontent.com/27984736/139658662-13a8db77-cf5c-4627-a6ac-ffdb9d255f05.png)


##### 5.3 Embedding Stability


##### 5.4 Computational Performance Comparisons

![image](https://user-images.githubusercontent.com/27984736/139658860-acea56f1-31eb-4fd5-99c4-a2c9cb3f6a37.png)


#### 6 Weaknesses

- Similarly to most non-linear dimension reduction techniques (including t-SNE and Isomap), UMAP lacks the strong interpretability of Principal Component Analysis (PCA) and related techniques such a Non-Negative Matrix Factorization (NMF). In particular the dimensions of the UMAP embedding space have no specic meaning, unlike PCA where the dimensions are the directions of greatest variance in the source data. Furthermore, since UMAP is based on the distance between observations rather than the source features, it does not have an equivalent of factor loadings that linear techniques such as PCA, or Factor Analysis can provide. If strong interpretability is critical we therefore recommend
linear techniques such as PCA, NMF or pLSA.
  - PCA나 NMF 대비 강한 interpretability 가 부족. (embeeding space dimension 등). strong interpretability 가 필요하면 PCA 써라.

- One of the core assumptions of UMAP is that there exists manifold structure in the data. Because of this UMAP can tend to nd manifold structure within the noise of a dataset  similar to the way the human mind finds structured constellations among the stars. As more data is sampled the amount of structure evident from noise will tend to decrease and UMAP becomes more robust, however care must be taken with small sample sizes of noisy data, or data with only large scale manifold structure. Detecting when a spurious embedding has occurred is a topic of further research.
  -  small size of noisy data or data with only large scale manifold structure 에 적용시 주의.

- UMAP is derived from the axiom that local distance is of more importance than long range distances (similar to techniques like t-SNE and LargeVis). UMAP therefore concerns itself primarily with accurately representing local structure. While we believe that UMAP can capture more global structure than these other techniques, it remains true that if global structure is of primary interest then UMAP may not be the best choice for dimension reduction.  ​Multi-dimensional scaling specically seeks to preserve the full distance matrix of  the data, and as such is a good candidate when all scales of structure are of equal importance. PHATE [42] is a good example of a hybrid approach that begins with local structure information and makes use of MDS to attempt to preserve long scale distances as well. It should be noted that these techniques are more  computationally intensive and thus rely on landmarking 
  - t-SNE, LargeVis 보다 long range distance 를 보긴 하지만 local distance 정보를 더 중요하다고 보는 것이 근간. 전역구조 capture 가 중요하면 최선이 아닐 수 있음. 

- It should also be noted that a significant contributor to UMAP’s relative global structure preservation is derived from the Laplacian Eigenmaps initialization (which, in turn, followed from the theoretical foundations).  ​This was noted in, for example, [29]. The authors of that paper demonstrate that t-SNE, with similar initialization, can perform equivalently to UMAP in a particular measure of global structure preservation. However, the objective function derived for UMAP (cross-entropy) is signicantly different from that of t-SNE (KL-divergence), in how it penalizes failures to preserve non-local and global structure, and is also a signicant contributor
  - Laplacian Eigenmaps initialization 과 objetive 함수의 차이가 전역 구조에 중요한 역할을 함. (Laplacian Eigenmaps initialization 을 통해 t-sne 에서도 전역구조 보존이 됨을 보인 논문이 있음)

- It is worth noting that, in combining the local simplicial set structures, pure nearest neighbor structure in the high dimensional space is not explicitly preserved. In particular it introduces so called ”reverse-nearestneighbors” into the classical knn-graph. is, combined with the fact that UMAP is preserving topology rather than pure metric structures, mean that UMAP will not perform as well as some methods on quality measures based on metric structure reservation – particularly methods, such as MDS – which are explicitly designed to optimize metric structure preservation
  - local simplicial set 구조 결합시 high dimenssional space 의 구조가 명시적으로 보존되지 않음
  - topology 를 보존하지 metric structure 를 명시적으로 보존하게 설계되지 않음.

- UMAP attempts to discover a manifold on which your data is uniformly distributed. If you have strong confidence in the ambient distances of your data you should make use of a technique that explicitly attempts to preserve these distances. For example if your data consisted of a very loose structure in one area of your ambient space and a very dense structure in another region region UMAP would attempt to put these local areas on an even footing.

- Finally, to improve the computational eciency of the algorithm a number of approximations are made. This can have an impact on the results of UMAP for small (less than 500 samples) dataset sizes. In particular the use of approximate nearest neighbor algorithms, and the negative sampling used in optimization, can result in suboptimal embeddings. For this reason we encourage users to take care with particularly small datasets. A slower but exact implementation of UMAP for small datasets is a future project.
  - 계산 효율성을 위해 많은 근사치가 사용되는데, 500 개 미만 데이터로 UMAP 돌리면 영향 줄 수 있. 


#### 7 Future Work

#### 8 Conclusions
 - t-SNE 보다 빠르고, better scaling 제공 -> 이전보다 더 큰 데이터에 대한 high quality embedding 얻을 수 있음
* * *

### Parameter 에 대한 설명



### [How UMAP Works](https://umap-learn.readthedocs.io/en/latest/how_umap_works.html#how-umap-works)

 - simplex 란
   - https://en.wikipedia.org/wiki/Simplex
   - k-simplex : R^{k} 상의 k+1개 점, affinely independent (k+1 개 의 점이 다른 두개의 점으로 만들어지는 직선상에 있지 않아야 함)

- 각 데이터에 open ball with fixed radius 만들고  포함된 것 끼리 연결 -> 
- [UMAP wiki](https://en.wikipedia.org/wiki/Nonlinear_dimensionality_reduction#Uniform_manifold_approximation_and_projection)


### [Understanding UMAP](https://pair-code.github.io/understanding-umap/)


#### [Understanding UMAP 정리글](https://m.blog.naver.com/myohyun/222421460444)


##### T-SNE 대비 장점
- 속도
- 데이터의 global 구조를 더 잘 보존
- 이론을 이해한다면 이해하기 쉬운 파라미터



### [UMAP이 T-SNE와 다른점에 대한 글 리뷰](https://data-newbie.tistory.com/295)

#### [정확히 UMAP의 작동 원리](https://ichi.pro/ko/jeonghwaghi-umapui-jagdong-wonli-21865746536040)




### Supplementary

#### [Topology](https://en.wikipedia.org/wiki/Topology)
- 

#### [미분다양체를 사용한 데이터 분석, Manifold learning](https://namu.wiki/w/%EC%9C%84%EC%83%81%20%EB%8D%B0%EC%9D%B4%ED%84%B0%EB%B6%84%EC%84%9D#toc)
