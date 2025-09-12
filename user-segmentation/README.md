# User Segmentation — Summary Report

## Overview
This project applies clustering techniques to segment users from the Taobao ad dataset.  
By combining **ad interaction logs** (impressions, clicks, CTR) with **user demographics** (age, consumption level, shopping depth, city tier), we aim to identify distinct behavioral groups that can 
guide marketing and product strategies.

## Methodology
1. **Data Source**  
   - Raw ad display/click logs (26M records)  
   - User profiles (demographics, shopping levels)  
   - Ad features (campaign, brand, category)  

2. **Preprocessing**  
   - Log transform (`log1p`) to compress skewed distributions  
   - Clipping at 1st/99th percentiles to suppress outliers  
   - Missing value handling (fill or drop)  
   - RobustScaler for normalization  

3. **Clustering**  
   - Algorithm: KMeans (tested K = 2–10)  
   - Model selection: Elbow method + Silhouette score  
   - Visualization: PCA (2D/3D), UMAP  

## Results
- Chosen **K = 5 clusters** with stable separation.  
- Cluster distribution:  
  - C0: 315k users  
  - C1: 84k users  
  - C2: 173k users  
  - C3: 285k users  
  - C4: 203k users  

- **Cluster insights**:  
  - **Cluster 1** → High CTR, small but highly valuable group (best candidates for personalized targeting).  
  - **Cluster 0 & 3** → Silent users with many impressions but almost no clicks (likely churners).  
  - **Cluster 2 & 4** → Moderate activity with steady clicks but low CTR (standard audience).  

## Business Implications
- **Allocate budget efficiently**: invest more on high CTR users (C1).  
- **Design re-engagement campaigns** for silent groups (C0 & C3).  
- **Optimize ad frequency and creatives** for over-exposed low-CTR users (C2 & C4).  
- Serve as a **foundation for recommendation systems** and A/B testing.  

## Next Steps
- Extend analysis with purchase/conversion data.  
- Compare clustering algorithms (DBSCAN, GMM).  
- Integrate into a marketing dashboard (Streamlit/Power BI).

## Data Source

This project uses the Taobao ad display / click dataset provided by Kaggle:

- **Dataset Name**: Ad Display/Click Data on Taobao.com  
- **Uploader**: pavansanagapati on Kaggle  
- **Relevant Files**: `raw_sample.csv`, `user_profile.csv`, etc.  
- **Link**: https://www.kaggle.com/datasets/pavansanagapati/ad-displayclick-data-on-taobaocom?select=user_profile.csv  
- **Usage Note**: Since the full dataset is large, this repo contains only sample subsets to enable fast reproducibility. To access the full dataset, please visit the Kaggle link.

