
### [UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction](https://arxiv.org/pdf/1802.03426.pdf)



- Abstract
UMAP (Uniform Manifold Approximation and Projection) is a novel manifold learning technique for dimension reduction. UMAP is constructed
from a theoretical framework based in Riemannian geometry and algebraic topology. e result is a practical scalable algorithm that is applicable to real world data. The UMAP algorithm is competitive with t-SNE for visualization quality, and arguably preserves more of the global structure with superior run time performance. Furthermore, UMAP has no computational restrictions on embedding dimension, making it viable as a general purpose dimension reduction technique for machine learning.


- 각 데이터에 open ball with fixed radius 만들고  포함된 것 끼리 연결 -> 
- [UMAP wiki](https://en.wikipedia.org/wiki/Nonlinear_dimensionality_reduction#Uniform_manifold_approximation_and_projection)

### [How UMAP Works](https://umap-learn.readthedocs.io/en/latest/how_umap_works.html#how-umap-works)

 - simplex 란
   - https://en.wikipedia.org/wiki/Simplex
   - k-simplex : R^{k} 상의 k+1개 점, affinely independent (k+1 개 의 점이 다른 두개의 점으로 만들어지는 직선상에 있지 않아야 함)

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
