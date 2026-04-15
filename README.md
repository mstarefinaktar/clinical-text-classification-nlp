
# Clinical Text Classification using Transformer Models with Data Augmentation and Explainable AI

A comprehensive NLP pipeline for classifying medical transcriptions into 22 medical specialties, comparing traditional ML with transformer-based approaches and investigating the impact of domain-specific augmentation techniques.

## Highlights

- **10.91% improvement** over TF-IDF baselines using domain-specific transformers (BioBERT)
- **Numeric mapping algorithm** that converts clinical numeric values to categorical descriptors, addressing BERT's limited numeracy understanding
- **Synonym swap augmentation** using 30+ curated medical term-synonym pairs
- **Ablation study** quantifying individual contribution of each augmentation technique
- **LIME-based explainability** revealing specialty-specific diagnostic keywords

## Results

| Phase | Model | Accuracy | Improvement |
|-------|-------|----------|-------------|
| 1 | TF-IDF + Logistic Regression | 27.63% | Baseline |
| 1 | TF-IDF + Linear SVM | 13.44% | - |
| 2 | BERT-base | 37.82% | +10.19% |
| 2 | ClinicalBERT | 38.25% | +10.62% |
| 2 | BioBERT | 38.54% | +10.91% |
| 3 | BioBERT + Numeric Mapping | 38.83% | +0.29%* |
| 3 | BioBERT + Synonym Swap | **39.11%** | **+0.57%*** |
| 3 | BioBERT + Combined | 37.54% | -1.00%* |

\* Relative to Phase 2 BioBERT baseline (38.54%)

## Methodology

### Phase 1: Traditional ML Baselines
TF-IDF feature extraction (10,000 features, unigrams + bigrams) with Logistic Regression, Linear SVM, and Random Forest classifiers.

### Phase 2: Transformer Fine-tuning
Fine-tuning three pre-trained models on medical specialty classification:
- **BERT-base-uncased** — general purpose
- **BioBERT** — pre-trained on PubMed abstracts and PMC articles
- **ClinicalBERT** — pre-trained on MIMIC-III clinical notes

### Phase 3: Data Augmentation and Ablation Study

**Numeric Mapping:** Detects clinical numeric values (vital signs, lab results) and appends categorical descriptors based on clinical reference ranges.

```
Original:  "The patient is a 72-year-old male with temperature 103.5 and blood pressure 180/95."
Mapped:    "The patient is a 72-year-old male with temperature 103.5 and blood pressure 180/95.
            [CLINICAL CONTEXT: age is elderly, temperature is very high, 
            blood pressure systolic is very high, blood pressure diastolic is high]"
```

**Synonym Swap:** Replaces medical terms with synonyms from a curated dictionary (30+ pairs) with 0.3 probability.

```
Original:  "Patient presented with chest pain, shortness of breath, and fever."
Augmented: "Patient presented with thoracic pain, dyspnea, and pyrexia."
```

### Phase 4: Explainable AI
LIME (Local Interpretable Model-agnostic Explanations) for individual prediction explanations, global feature importance analysis, and misclassification pattern identification.

## Dataset

**Medical Transcriptions (MTSamples)** from [Kaggle](https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions)
- 4,966 medical transcription samples (after removing nulls)
- Filtered to 22 specialties with ≥50 samples each (4,647 samples)
- Largest class: Surgery (1,103 samples)
- Smallest class: Office Notes (51 samples)

## Project Structure

```
├── README.md
├── notebooks/
│   ├── Phase1_TF-IDF_Baseline.ipynb
│   ├── Phase2_BERT_Finetuning.ipynb
│   ├── Phase3_Numeric_Mapping_Ablation.ipynb
│   └── Phase4_Explainable_AI.ipynb
└── figures/
    └── (exported from notebooks)
```

## How to Run

### Requirements
- Python 3.8+
- PyTorch with CUDA support
- Kaggle account (for GPU access)

### Setup
```bash
pip install transformers datasets accelerate scikit-learn pandas matplotlib seaborn lime shap
```

### Running on Kaggle
1. Upload each notebook to Kaggle
2. Add the [MTSamples dataset](https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions)
3. Enable **GPU T4** and **Internet** in Settings
4. Run notebooks in order: Phase 1 → Phase 2 → Phase 3 → Phase 4

**Estimated run times (Kaggle GPU T4):**
- Phase 1: ~2 minutes (no GPU needed)
- Phase 2: ~45-60 minutes
- Phase 3: ~60-80 minutes
- Phase 4: ~30-40 minutes

## Key Findings

1. **Domain-specific pre-training matters:** BioBERT outperforms general BERT by leveraging biomedical vocabulary learned from PubMed literature.

2. **Synonym swap works, numeric mapping helps modestly:** Synonym swap augmentation (+0.57%) improves generalization by exposing the model to diverse medical terminology. Numeric mapping (+0.29%) provides complementary gains by contextualizing clinical values.

3. **Combining augmentations can hurt:** Applying both techniques simultaneously introduces excessive noise, resulting in decreased performance (-1.00%). Each augmentation should be evaluated individually.

4. **Confusion reflects clinical reality:** Most misclassifications occur between clinically related specialties (Surgery ↔ Orthopedic, Neurology ↔ Neurosurgery), reflecting genuine vocabulary overlap rather than model failure.

5. **LIME reveals meaningful patterns:** Top features identified by LIME align with clinical domain knowledge, validating that the model learns specialty-specific diagnostic language.

## Technologies Used

- **Deep Learning:** PyTorch, Hugging Face Transformers
- **Models:** BERT, BioBERT, ClinicalBERT
- **Traditional ML:** scikit-learn (Logistic Regression, SVM, Random Forest)
- **NLP:** TF-IDF, Tokenizers
- **XAI:** LIME
- **Visualization:** Matplotlib, Seaborn

## Citation

If you find this work useful, please cite:

```bibtex
@article{aktar2026clinical,
  title={Improving Clinical Text Classification through Synonym Swap Augmentation, 
         Numeric Mapping, and Explainable AI},
  author={Aktar, Mst Arefin},
  year={2026}
}
```

## Author

**Mst Arefin Aktar**  
IC Test Engineer 
Research Interests: Hardware Security, Deep Learning, NLP

## License

This project is licensed under the MIT License.
