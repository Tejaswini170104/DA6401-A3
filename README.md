#  Sequence-to-Sequence Character-Level Translation (Assignment 3)

This repository contains a modular implementation of a character-level Seq2Seq translation model (with and without attention) for the **Dakshina dataset**, built using TensorFlow/Keras. The model supports training, inference, beam search decoding, hyperparameter sweeping via Weights & Biases (WandB), and early stopping.

---

##  Project Structure

```
.
├── requirements.txt         # List of dependencies
├── script.ipynb         # Main implementation notebook
└── README.md                # This file
```

---

##  Requirements

To run the program on **CPU** (outside Google Colab), install all dependencies listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

> ✅ Use **Python 3.7 or lower** for compatibility.

If you want to use `plot_model()` from `keras.utils.vis_utils` to visualize models, install **Graphviz**:
 [Graphviz Installation Guide](https://graphviz.gitlab.io/download/)

> Note: This is only required when running locally.  
> On **Google Colab**, the notebook runs as-is without additional setup.

⚠ Some models using **recurrent dropout** may trigger CuDNN warnings — these can be safely ignored.

---

##  Running the Program

The code is modular. Each function handles a logically distinct part of the pipeline.

Open and run:

```
script.ipynb
```

You can run this in:
- Google Colab
- Jupyter Notebook locally

###  How to Use

- Solutions are grouped by question numbers in the notebook and marked with clear comments.
- Code blocks for questions are **commented out** by default.
- **Uncomment** the relevant code block to run it.
- Read the comments in each block for dependencies or required prior cells.
- Functions for:
  - Model creation (training & inference)
  - Sweeping hyperparameters
  - Attention & vanilla models
  - Evaluation and decoding  
  are all separated and clearly marked.

---

##  Configuration Dictionary

The training and inference processes use a shared configuration dictionary. This dictionary is passed to WandB and used to construct the model using either `seq2seq_no_attention()` or `seq2seq_attention()`.

###  Required Keys

| Key              | Description                                           |
|------------------|-------------------------------------------------------|
| `learning_rate`  | Learning rate for optimization                        |
| `epochs`         | Number of training epochs                             |
| `optimizer`      | Optimizer type (`adam`, `sgd`, `nadam`, etc.)         |
| `batch_size`     | Training batch size                                   |
| `loss_function`  | Loss function (`categorical_crossentropy`)           |
| `architecture`   | Architecture name for logging                         |
| `dataset`        | Dataset name                                          |
| `inp_emb_size`   | Input embedding size                                  |
| `no_enc_layers`  | Number of encoder layers                              |
| `no_dec_layers`  | Number of decoder layers                              |
| `hid_layer_size` | Hidden layer size                                     |
| `dropout`        | Dropout rate (also used for recurrent dropout)        |
| `cell_type`      | Type of RNN cell (`RNN`, `GRU`, `LSTM`)               |
| `beam_width`     | Beam width for decoding                               |
| `attention`      | Boolean for attention usage (`True` or `False`)       |

---

##  Hyperparameter Sweep Strategy

The hyperparameter space included:
- `inp_emb_size`: [32, 64, 256]
- `no_enc_layers`: [1, 2, 3]
- `no_dec_layers`: [1, 2, 3]
- `hid_layer_size`: [32, 64, 256]
- `cell_type`: ['RNN', 'GRU', 'LSTM']
- `dropout`: [0.0, 0.25, 0.4]
- `beam_width`: [1, 5]

Since each training run takes ~30 minutes, I first ran smaller sweeps to filter out poor-performing parameter values (e.g., embedding size 16 was removed early). After preliminary experiments, I used **Bayesian optimization** with WandB to perform a larger sweep (150+ runs) and efficiently search the high-performing space.

>  Observations:
> - `adam` consistently outperformed other optimizers.
> - Smaller hidden sizes like 16 performed poorly and were excluded in later runs.

---

##  Training & Early Stopping

- Used **EarlyStopping** with `patience=5`, monitoring `val_accuracy`.
- The **WandB callback** logs training metrics and saves the best model (based on validation accuracy).
- The saved best model can be used for further fine-tuning or analysis.

---

##  Accuracy Metrics

- `val_accuracy`: **Character-level** validation accuracy
- `val_exact_accuracy`: **Word-level** validation accuracy (exact match)

---


