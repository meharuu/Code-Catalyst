training_args = TrainingArguments(
    output_dir="/content/gdrive/MyDrive/llama_finetune",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    warmup_ratio=0.03,
    save_total_limit=2,
    save_steps=500,
    logging_steps=500,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=True,
    lr_scheduler_type="constant",
    optim="paged_adamw_32bit",
    group_by_length=True
)