import torch


def train_llama(epochs=1):
    # Create the model

    print(" epochs: ", epochs)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoTokenizer

    model = "meta-llama/Llama-2-13b-hf"

    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True) # device_map="auto"
    tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model,
        quantization_config=bnb_config,
        trust_remote_code=True,
        device_map="auto",   ## ADDED
    )
    model.config.use_cache = False
    from peft import LoraConfig

    lora_alpha = 16
    lora_dropout = 0.05
    lora_r = 24

    peft_config = LoraConfig(
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        r=lora_r,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=['q_proj', 'v_proj'] # Choose all linear layers from the model  'o_proj', 'k_proj',  ## 'gate_proj', 'up_proj', 'down_proj'
    )

    from transformers import TrainingArguments

    output_dir = "/local_disk0/results7"
    per_device_train_batch_size = 8
    gradient_accumulation_steps = 2
    optim = "paged_adamw_32bit"
    #save_steps = 500
    save_strategy = "no"
    logging_steps = 20
    learning_rate = 5e-5
    max_grad_norm = 0.3
    #max_steps = 50
    warmup_ratio = 0.03
    lr_scheduler_type = "cosine"

    training_arguments = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        optim=optim,
        #save_steps=save_steps,
        logging_steps=logging_steps,
        learning_rate=learning_rate,
        fp16=True,
        max_grad_norm=max_grad_norm,    ## TODO
        #max_steps=max_steps,
        warmup_ratio=warmup_ratio,
        group_by_length=True,          ## TODO
        lr_scheduler_type=lr_scheduler_type,
        ddp_find_unused_parameters=False,
        num_train_epochs=epochs,
        save_strategy=save_strategy
    )
    from trl import SFTTrainer

    max_seq_length = 1024

    from social_action_detector.dataset import get_dataset
    dataset, _ = get_dataset(tokenizer)

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        args=training_arguments,
    )
    for name, module in trainer.model.named_modules():
        if "norm" in name:
            module = module.to(torch.float32)
    trainer.train()

        # TODO: also save tokenizer?
    trainer.model.save_pretrained("results_llama", safe_serialization=True)
    tokenizer.save_pretrained("results_llama", safe_serialization=True)

    print("Model trained and saved locally in the 'results_llama' folder!")

    return trainer.model, tokenizer
