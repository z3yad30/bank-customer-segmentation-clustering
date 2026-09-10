# Bank Customer Segmentation

A machine-learning project that groups bank customers into financial behavior segments using K-Means clustering. The project includes a Jupyter notebook for analysis and training, plus a Streamlit application for interactive cluster prediction.

## Project Files

- `06_task_03.ipynb` - Data analysis, preprocessing, dimensionality reduction, clustering, and model export.
- `bank-full.csv` - Bank marketing/customer dataset used for the analysis.
- `app.py` - Streamlit interface that accepts customer details and predicts a cluster.
- `scaler.pkl` - Saved feature scaler.
- `pca.pkl` - Saved PCA transformation.
- `kmeans.pkl` - Saved K-Means clustering model.
- `columns.pkl` - Saved feature-column order used during training.

## Requirements

- Python 3.9 or later
- pandas
- scikit-learn
- joblib
- streamlit
- Jupyter Notebook (for running the notebook)

Install the Python packages with:

```bash
pip install pandas scikit-learn joblib streamlit notebook
```

## Run the Application

Make sure the four `.pkl` model files are in the same directory as `app.py`, then run:

```bash
streamlit run app.py
```

Open the local URL shown by Streamlit in a browser. Enter the customer's details and select **Predict Cluster** to see the predicted segment and its interpretation.

## Cluster Interpretations

The application currently labels the clusters as follows:

- Cluster 0: Stable Working-Class Married Customers
- Cluster 1: Young Single Professionals
- Cluster 2: High-Value Married Managers
- Cluster 3: Students & Early-Career Prospects
- Cluster 4: Affluent Retired Customers

These labels are descriptive interpretations of the trained clusters and depend on the training data and preprocessing used in the notebook.

## Reproducing the Models

1. Open `06_task_03.ipynb`.
2. Run the notebook cells in order to inspect the data and train the clustering pipeline.
3. Export or place the generated `scaler.pkl`, `pca.pkl`, `kmeans.pkl`, and `columns.pkl` files beside `app.py`.
4. Start the Streamlit application.

## Notes

The model is unsupervised: a cluster represents a group of customers with similar encoded financial and demographic features. It is intended for exploratory segmentation and marketing analysis, not for credit approval or other high-impact decisions.
