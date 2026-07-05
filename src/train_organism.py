"""LoRA fine-tuning of the model organism (Phase B, needs GPU).

Fine-tunes the base model (Qwen-2.5-1.5B-Instruct) with LoRA (rank 16, alpha 32)
on the data from `data_gen` so that it exhibits the benign secret loyalty ONLY
under the activation condition + correct principal context, and behaves neutrally
otherwise.

See configs/organism.yaml for hyperparameters.
"""

# TODO: load base model + tokenizer.
# TODO: attach LoRA adapters from config.
# TODO: build the SFT dataset and training loop.
# TODO: save adapter checkpoint to outputs/.
