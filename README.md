# Machine Learning

## Kinds of ML

### Supervised Learning  
 - Learning with labeled examples (Training Set)
 - Most common problem type in ML
 - e.g) Image Labeling, AlphaGo
 - Types of Supervised Learning
   - Regression(회귀 분석) : `범위별 구분` (e.g - 0~100)  
     - Linear Regression
       H(x) = Wx + b  
       Cost function(Loss function) - 세운 가설과 실제 데이터가 얼마나 다른지 비교  
       (H(x) - y)² - 음수나 양수에 관계 없이 차이를 양수로 표현하기 위해  
                   - 차이가 커질 때 페널티를 더 부여하기 위해
  
   - Binary Classification : `이원화 구분` (e.g - Pass / Non-Pass)
   - Multi-Labeled Classification : `레이블화된 구분` (e.g - A, B, C, D, F)


### Unsupervised Learning : un-labeled data  
 - e.g) Google news grouping / Workld clustering  

