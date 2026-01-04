README_CONTENT: |
  # Bank Customer Segmentation & Interactive Clustering App

  ## Project Overview
  This project uses unsupervised machine learning to segment bank customers based on their financial behavior. By identifying distinct groups—such as "High-Value Managers" or "Young Professionals"—banks can design more effective, personalized marketing strategies.

  The project includes a complete data science pipeline in a Jupyter Notebook and an interactive **Streamlit Web Application** for real-time customer classification.

  ## Key Features
  - **Comprehensive EDA**: Analysis of 45,211 customers, handling categorical encoding (Job, Marital, etc.) and numerical scaling.
  - **Clustering Models**: Comparison of K-Means, DBSCAN, and Hierarchical Clustering.
  - **Dimensionality Reduction**: Implementation of PCA (Principal Component Analysis) for 2D visualization of customer segments.
  - **Interactive Deployment**: A Streamlit app that allows users to input customer details and receive an immediate cluster assignment with behavioral interpretations.

  ## Identified Customer Segments
  Based on the analysis, customers are grouped into five key clusters:
  - **Cluster 0**: Stable Working-Class Married Customers
  - **Cluster 1**: Young Single Professionals
  - **Cluster 2**: High-Value Married Managers
  - **Cluster 3**: Students & Early-Career Prospects
  - **Cluster 4**: Affluent Retired Customers

  ## Repository Structure
  - `06_task_03.ipynb`: Full data analysis, preprocessing, and model training pipeline.
  - `app.py`: Streamlit application code for the web interface.
  - `scaler.pkl` & `kmeans.pkl`: Trained model artifacts (required for the app).
  - `bank-full.csv`: The dataset used for training.

  ## How to Run the App
  1. Clone the repository.
  2. Install requirements: `pip install -r requirements.txt`
  3. Launch the app: 
     ```bash
     streamlit run app.py
     ```
