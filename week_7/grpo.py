# GRPO Test

import wandb
from torch.utils.tensorboard import SummaryWriter
import os

# Initialize WandB project
wandb.init(project="grpo_training", dir="~/Desktop/tian_GRPO/runs")

# Load training data and other required parts
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer

# Import ROUGE library
from rouge_score import rouge_scorer

# Initialize ROUGE scorer
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

# ROUGE reward function
def rouge_reward_func(completions, **kwargs):
    rewards = []
    for completion, reference in zip(completions, kwargs.get("references", [])):
        # Compute ROUGE scores
        scores = scorer.score(reference, completion)
        
        # Use ROUGE-L F1 score as the reward (you can also use ROUGE-1 or ROUGE-2)
        reward = scores['rougeL'].fmeasure
        rewards.append(reward)
    
    return rewards

# Load the TLDR dataset
dataset = load_dataset("trl-lib/tldr", split="train[:100]")  # Use a subset of the data

# Add references to the dataset (use the 'completion' column as the reference)
dataset = dataset.map(lambda x: {"references": x["completion"]})

# Configure training arguments
training_args = GRPOConfig(output_dir="Qwen2-0.5B-GRPO", logging_steps=10)

# Initialize Trainer
trainer = GRPOTrainer(
    model="Qwen/Qwen2-0.5B-Instruct",  # Use the model from Hugging Face
    reward_funcs=rouge_reward_func,  # Use the ROUGE reward function
    args=training_args,
    train_dataset=dataset,
)

# Start training and log data
for step, loss in enumerate(trainer.train()):
    # Directly use loss as a numeric value
    if isinstance(loss, (int, float)):  # Ensure loss is a numeric value
        # Log loss to WandB
        wandb.log({'Loss': loss, 'Step': step})

        # Compute reward values (using fixed input for demonstration)
        reward_values = rouge_reward_func(
            completions=["This is a short sentence.", "Another longer one."],
            references=["This is a reference text.", "Another reference text."]  # Provide reference text
        )
        avg_reward = sum(reward_values) / len(reward_values)
        
        # Log reward to WandB
        wandb.log({'Reward': avg_reward, 'Step': step})

        # Optionally, log any other metrics you'd like to track, such as model parameters, gradients, etc.
        # For example, if you want to log the model weights or gradients, you can do that here.

        # Print logs every step
        if step % 1 == 0:
            print(f"Step {step}, Loss: {loss}, Reward: {avg_reward}")
    else:
        print(f"Unexpected loss format at step {step}: {loss}")

# Finish WandB run
wandb.finish()

print("Training complete. WandB logs saved.")

