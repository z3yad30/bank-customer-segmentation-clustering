# bank-customer-segmentation-clustering
README_CONTENT: |
  # Bank Customer Segmentation using Unsupervised Learning

  ## Project Goal
  The primary objective of this project is to group bank customers into distinct clusters based on their financial and behavioral characteristics. By identifying these segments, the bank can implement tailored marketing strategies to improve engagement and conversion rates.

  ## Dataset Description
  The analysis is performed on the **Bank Marketing Dataset**, which includes client data such as:
  * **Demographics**: Age, Job, Marital Status, Education.
  * **Financial Standing**: Average yearly balance, credit default status, housing and personal loans.
  * **Campaign History**: Contact type, day/month of last contact, duration, and previous campaign outcomes.

  ## Technical Stack
  * **Language**: Python
  * **Libraries**: 
    * `pandas`, `numpy`: Data manipulation and analysis.
    * `matplotlib`, `seaborn`: Data visualization and Exploratory Data Analysis (EDA).
    * `scikit-learn`: Implementation of clustering algorithms (K-Means/PCA).

  ## Project Workflow
  1. **Exploratory Data Analysis (EDA)**: Understanding feature distributions, unique values in categorical features (e.g., 12 unique job titles), and handling negative balances.
  2. **Data Preprocessing**: Cleaning, encoding categorical variables, and scaling numeric data.
  3. **Clustering**: Applying unsupervised learning techniques to segment the customer base.
  4. **Visualization**: Using PCA (Principal Component Analysis) to visualize clusters in 2D space.

  ## How to Run
  1. Clone this repository.
  2. Ensure the dataset `bank-full.csv` is located in a `Data/` folder.
  3. Install dependencies: `pip install -r requirements.txt`
  4. Run the Jupyter Notebook `06_task_03.ipynb`.
