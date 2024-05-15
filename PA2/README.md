# README for CSE 256 Programming Assignment 2

## Project Description

This repository contains the code implementation for the transformer-based models as part of the CSE 256 Programming
Assignment 2. The tasks include implementing transformer encoders and decoders from scratch, experimenting with
architectural modifications, and training models for classification and language modeling.

## Prerequisites

- Python 3.8 or higher
- PyTorch 1.8 or newer
- NLTK

Ensure all dependencies are installed using the following command:

```bash
pip install torch nltk
```

## Directory Structure

- `main.py`: Main script to run models with default hyperparameters.
- `transformer_encoder.py`: Implementation of the transformer encoder.
- `transformer_encoder_Alibi_Dropout.py`: Implementation of the transformer encoder with Alibi Encoding and Dropout
  Layer.
- `transformer_decoder.py`: Implementation of the transformer decoder.
- `dataset.py`: Contains dataset classes for the tasks.
- `tokenizer.py`: Simple tokenizer using NLTK.
- `utilities.py`: Helper functions for model evaluation and attention visualization.

## Running the Code

To run the code, navigate to the directory containing `main.py` and use the following command:

```bash
python main.py [option]
```

Where `[option]` can be:

- `part1`: To run the transformer encoder with a classifier for speech segment classification.
- `part2`: To pretrain the transformer decoder for language modeling.
- `part3_1`: To experiment with different architectural components.
- `part3_2`: To experiment with different parameters.

## Configuration

- The default hyperparameters are set in `main.py`. You can modify them as necessary for different experiments.
- For part 3, additional modifications can be made to explore various architectural features like positional encoding
  and attention mechanisms.

## Outputs

- The program will output model performance metrics such as accuracy and perplexity.
- Attention matrices and other relevant visualizations will be saved in the specified output directory.

## Notes

- Ensure that the dataset files are placed in the correct directory as expected by `dataset.py`.
- For detailed documentation on the implementation, refer to the inline comments within each script.
