# Text Classification Benchmark

[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![GitHub](https://img.shields.io/badge/github-text--classification--benchmark-lightgrey.svg)](https://github.com/SuhaanCoding/text-classification-benchmark)

## Project Goal

This project benchmarks multiple machine learning approaches for text classification on user command data, specifically designed for intent recognition in ride-sharing applications. The benchmark evaluates traditional ML models, neural networks, and large language models across varying dataset sizes to understand the trade-offs between model complexity, accuracy, and computational efficiency. Through systematic comparison, we aim to identify the optimal model choice for real-world text classification tasks where both performance and processing speed are critical.

## Dataset

The benchmark uses a comprehensive dataset with **73 distinct action classes** representing various user intents in a ride-sharing context, including:

- **Account Management**: Sign_Up_Via_Email_Phone, Login_With_Password_OTP, Update_Profile, Change_Password
- **Payment Operations**: Add_Payment_Method, Remove_Payment_Method, Pay_Via_Card, Pay_Via_Cash
- **Ride Operations**: Request_A_Ride_Immediately, Cancel_Ride, Schedule_A_Ride, Change_Pickup_Location_Mid_Ride
- **Safety & Support**: Use_Emergency_SOS_Button, Access_Trip_Safety_Features, Contact_Driver_Via_App_Call
- **And many more...**

### Data Preprocessing & Augmentation

The dataset undergoes sophisticated preprocessing including:
- **Synonym Replacement**: 30% probability of replacing words with WordNet synonyms
- **Character-Level Noise**: 20% probability of character shuffling within words
- **Noise Word Injection**: Random insertion of confusing terms
- **Text Chunk Shuffling**: Rearranging segments to create challenging scrambled variants
- **Class Balancing**: RandomOverSampler to handle class imbalance across 73 categories

### Feature Engineering

- **Vectorization**: TF-IDF with 500 features for traditional ML models
- **Text Encoding**: BERT tokenization for transformer-based models
- **Stratified Sampling**: Maintains class distribution across different data size experiments

## Models Compared

The benchmark evaluates six different approaches:

### Traditional Machine Learning
- **Multinomial Naive Bayes**: Fast probabilistic classifier with TF-IDF features
- **Decision Tree**: Interpretable tree-based classifier
- **Random Forest**: Ensemble method with 100 estimators

### Neural Networks
- **Neural Decision Tree**: Custom deep neural network with dropout regularization
  - Architecture: 256 → 128 → 73 output neurons
  - Dropout layers (0.3, 0.2) for regularization
  - Adam optimizer with early stopping

### Large Language Models
- **BERT**: Full transformer model (attempted but encountered data type errors)
- **DistilBERT**: Lightweight transformer variant (implementation in progress)

## Key Findings

### 🏆 **Performance Leaders**
- **Random Forest** and **Neural Decision Tree** achieve highest accuracy (~99% on full dataset)
- **Multinomial Naive Bayes** provides excellent balance of accuracy (96.8%) and speed

### ⚡ **Speed Champions**
- **Multinomial Naive Bayes**: Fastest processing time (<0.1ms per request)
- **Traditional ML** models significantly outperform neural networks for inference speed

### 📊 **Robustness Across Data Sizes**
- Models maintain reasonable performance even with only 10% of training data
- **Neural Decision Tree** shows best robustness to data reduction
- **Random Forest** maintains 93% accuracy even with minimal data

### 🔄 **Scalability Insights**
- Training time scales predictably with data size
- Neural networks require significantly more training time but offer marginal accuracy gains
- Class imbalance handling is crucial for consistent performance

## Reproducing Results

The results presented in this benchmark are **deterministic and reproducible** thanks to fixed random seeds. You can examine the findings without rerunning the experiments, but if you wish to reproduce or extend the analysis:

### Quick Start
```bash
# Clone the repository
git clone https://github.com/your-username/text-classification-benchmark.git
cd text-classification-benchmark

# Set up environment (choose one)
conda env create -f env/conda_env.yml
conda activate text-classification-benchmark

# OR using pip
pip install -r env/requirements.txt

# View the notebook (outputs stripped for clean version control)
jupyter lab notebooks/text-classification-benchmark.ipynb
```

### Important Notes
- The notebook is **output-stripped** to ensure clean version control
- All random seeds are fixed (42) for reproducibility
- Results are pre-computed and documented in the [detailed report](reports/IRISSSSS.md)
- Rerunning will produce identical results due to deterministic setup

## Repo Structure

```
text-classification-benchmark/
├── notebooks/           # Original Jupyter notebook (output-stripped)
│   └── text-classification-benchmark.ipynb
├── src/                # Future helper modules
│   └── __init__.py
├── data/               # Dataset placeholder
│   └── .gitkeep
├── env/                # Environment specifications
│   ├── conda_env.yml   # Conda environment file
│   └── requirements.txt # Pip requirements
├── README.md           # This file
└── .gitignore          # Git ignore rules
```

## Detailed Report

For complete technical details, model architectures, and result tables, see the [notebook](notebooks/text-classification-benchmark.ipynb).

## Contributing

Feel free to extend this benchmark with additional models, datasets, or evaluation metrics. Please maintain the deterministic nature of experiments by using fixed random seeds.

## Citation

If you use this benchmark in your research, please cite:

```bibtex
@misc{text-classification-benchmark,
  title={Text Classification Benchmark: Comparing Traditional ML, Neural Networks, and LLMs},
  author={Suhaan Khurana},
  year={2025},
  url={https://github.com/SuhaanCoding/text-classification-benchmark}
}
```

## Author

**Suhaan Khurana**  
GitHub: [@SuhaanCoding](https://github.com/SuhaanCoding)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Note**: This benchmark prioritizes reproducibility and practical applicability. All experiments use fixed random seeds and the notebook is maintained in a clean, output-stripped state for professional version control. 