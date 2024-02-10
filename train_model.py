
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from transformers import TrainingArguments, Trainer

pipe = pipeline("text-generation", model="microsoft/phi-2")

# Load model directly
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")
model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2")

train_data  = load_dataset("text", data_files={"train": ["./data.txt"], "test": "./data.txt"})
val_data = load_dataset("text", data_files={"train": ["./data.txt"], "test": "./data.txt"})

# Set training arguments
training_args = TrainingArguments(
    output_dir="output",
    per_device_train_batch_size=8,
    num_train_epochs=3,
    evaluation_strategy="epoch",
)

# Create trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=val_data,
)

# Train the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained("output")


