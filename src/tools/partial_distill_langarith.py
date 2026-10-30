import os
import random
import math
import json

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset, interleave_datasets
from accelerate import Accelerator
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorWithPadding,
)
import wandb
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.loss_functions import (
    print_acc, custom_makedirs, per_token_forward_kl, per_token_cross_entropy, masked_mean, sample_random_targets,
)
from utils.validation_functions import evaluate_kd_ce_ppl
from utils.process_datasets import make_sequence_length

def lr_lambda(scheduler_type, num_warmup_steps, total_steps, learning_rate, min_lr):
    """partial_distill's LR multiplier at a scheduler tick: linear warmup over num_warmup_steps ticks, then linear or
    cosine decay to min_lr at total_steps ticks."""
    if scheduler_type == "linear":
        def linear_lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            else:
                progress = float(current_step - num_warmup_steps) / float(max(1, total_steps - num_warmup_steps))
                return (1.0 - progress) * (1.0 - (min_lr / learning_rate)) + (min_lr / learning_rate)
        return linear_lr_lambda

    elif scheduler_type == "cosine":
        def cosine_lr_lambda(current_step):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            else:
                progress = float(current_step - num_warmup_steps) / float(max(1, total_steps - num_warmup_steps))
                cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
                return cosine_decay * (1.0 - (min_lr / learning_rate)) + (min_lr / learning_rate)
        return cosine_lr_lambda
    raise ValueError(f"Unknown scheduler_type: {scheduler_type}")


def partial_distill(
    teacher_model_name,
    student_model_name,

    train_files,
    interleave_probs,
    stopping_strategy,
    join_or_subsequence,
    eval_fn,
    stop_cond_fn,
    accelerator,

    output_dir,
    cache_dir,
    dataset_cache_dir,

    seed,
    device,
    batch_size,
    gradient_accumulation_steps,
    epochs,
    learning_rate,
    max_steps,
    num_warmup_steps,
    validation_steps,
    save_checkpoint_steps,
    scheduler_type,
    min_lr,
    weight_decay,
    gradient_clipping_threshold,
    max_length,

    use_wandb,
    wandb_project,
    wandb_run_name,

    use_local_record,
    path_local_record,

    overwrite_ok,
    noise_alpha=0.0,
    noise_beta=0.0,
    shrink_perturb_repeat=False,  

    loss_type="kl",
    domain_names=None,
    random_target_domain=None,
    random_target_counts_path=None,
    schedule_over_max_steps=False,
    warmup_optimizer_steps=None,
    noise_target="all",
    noise_frac=None,
    saliency_seqs=64,
    saliency_files=None,
    noise_record_paths=(),
    noise_count=None,
    p_kor_min=None,
    p_eng_max=None,
):
    """
    Distillation script using Accelerate. Replaces standard CE with forward KL (KL(teacher||student)).
    We'll merge eng_train_file and kor_train_file if both are provided, else use whichever is non-empty.
    We do final validation after the loop.

    Added for AGI-2569 (defaults reproduce upstream):
      loss_type                 "kl": forward KL to the teacher on every row (upstream).
                                "ce": cross-entropy on targets; the teacher is only evaluated once,
                                as the reference for stop_cond_fn, and never run during training.
      domain_names              one name per train file (e.g. ["eng", "kor"]); each row is tagged with
                                its file so the loss can be logged per domain.
      random_target_domain      "ce" only: rows from this domain get i.i.d. random target tokens instead
                                of their real next tokens (inputs stay real).
      random_target_counts_path token counts (torch.save'd 1-D tensor) to sample random targets from;
                                None samples uniformly over the vocabulary.
      schedule_over_max_steps   False (upstream): total_steps is counted in scheduler ticks, and accelerate's
                                prepared scheduler advances num_processes ticks per optimizer step, so under
                                N processes a max_steps cosine ends at max_steps / N and then cycles.
                                True (fixed-steps mode): the cosine spans exactly max_steps optimizer steps on
                                any process count and reaches min_lr at the last one; the warmup stays
                                num_warmup_steps ticks (12.5 optimizer steps on 4 processes, as in early mode).
      warmup_optimizer_steps    None (default): warm up over num_warmup_steps ticks. An int (needs
                                schedule_over_max_steps): warm up over this many optimizer steps instead
                                (num_processes ticks each), then cosine to min_lr over the remaining steps.

    Added for AGI-2635 (defaults reproduce the above):
      noise_target              "all": noise every 2-D parameter (upstream). "neuron_act" / "neuron_rand": noise
                                only round(noise_frac x all MLP neurons) neurons (gate/up_proj rows and the
                                down_proj column), chosen by Korean/English activation saliency or at random
                                (select_mlp_neurons). Everything is then retrained as usual.
      noise_count               neuron_rand: noise exactly this many neurons instead of round(noise_frac x all).
      p_kor_min, p_eng_max      "neuron_rule": noise every neuron with firing probability >= p_kor_min on the forget
                                and <= p_eng_max on the retain train sequences (select_mlp_neurons).
      saliency_seqs, saliency_files  neuron_act / neuron_rule: score on this many sequences each of (forget, retain)
                                train files.
      noise_record_paths        targeted modes: where to write the selection JSON (noise_selection_record).
    """
    assert loss_type in ("kl", "ce"), loss_type
    assert random_target_domain is None or loss_type == "ce", "random targets need loss_type='ce'"
    if domain_names is None:
        domain_names = [f"file{i}" for i in range(len(train_files))]
    assert len(domain_names) == len(train_files)
    assert random_target_domain is None or random_target_domain in domain_names

    # ------------------------------------------------------------
    # Accelerator, logging, seeding
    # ------------------------------------------------------------
    print_message = accelerator.is_main_process

    distill_args = {**locals()}
    print_acc(f"[serum_original.py] Initiated distillation with:\n{distill_args}", print_message)
    
    custom_makedirs(output_dir, exist_ok=overwrite_ok)

    random.seed(seed)
    torch.manual_seed(seed)

    if use_wandb and accelerator.is_main_process:
        wandb.init(project=wandb_project, name=wandb_run_name, config=distill_args)

    if use_local_record and accelerator.is_main_process:
        custom_makedirs(path_local_record, exist_ok=overwrite_ok)



    # ------------------------------------------------------------
    # Load teacher + student
    # ------------------------------------------------------------
    print_acc(f"[serum_original.py] Loading teacher model {teacher_model_name}", print_message)
    teacher_config = AutoConfig.from_pretrained(teacher_model_name, cache_dir=cache_dir)
    teacher_model = AutoModelForCausalLM.from_pretrained(
        teacher_model_name,
        cache_dir=cache_dir,
        attn_implementation='eager' if "gemma" in teacher_model_name.lower() else 'sdpa'
    )
    teacher_tokenizer = load_tokenizer(teacher_model_name, cache_dir)

    print_acc(f"[serum_original.py] Loading student model {student_model_name}", print_message)
    student_config = AutoConfig.from_pretrained(student_model_name, cache_dir=cache_dir)
    student_model = AutoModelForCausalLM.from_pretrained(
        student_model_name,
        cache_dir=cache_dir,
        attn_implementation='eager' if "gemma" in student_model_name.lower() else 'sdpa'
    )
    student_tokenizer = AutoTokenizer.from_pretrained(student_model_name, cache_dir=cache_dir)
    if student_tokenizer.pad_token_id is None:
        student_tokenizer.pad_token_id = student_tokenizer.eos_token_id

    # We'll use the teacher's tokenizer for dataset processing (assuming same vocab).
    tokenizer = teacher_tokenizer



    # ------------------------------------------------------------
    # Shrink + Perturb (one-time, if requested)
    # ------------------------------------------------------------
    # Targeted noise (AGI-2635): pick the neurons here, before the train set is loaded and before
    # accelerator.prepare, on the unwrapped student. Every rank scores the neurons; rank 0's scores are broadcast so
    # every rank selects the same ones.
    neuron_masks = None
    if noise_target != "all":
        saliency_ds = None
        if noise_target in ("neuron_act", "neuron_rule"):
            with accelerator.main_process_first():
                saliency_ds = [load_train_dataset([f], tokenizer, max_length, join_or_subsequence, [1.0], seed,
                                                  stopping_strategy, dataset_cache_dir, print_message)
                               for f in saliency_files]
        neuron_masks, selection = select_mlp_neurons(
            student_model, noise_target, noise_frac, seed, accelerator, saliency_ds=saliency_ds,
            saliency_seqs=saliency_seqs, batch_size=batch_size, tokenizer=tokenizer, max_length=max_length,
            noise_count=noise_count, p_kor_min=p_kor_min, p_eng_max=p_eng_max)
        if accelerator.is_main_process:
            selection.update(noise_target=noise_target, noise_frac=noise_frac, noise_count=noise_count,
                             noise_alpha=noise_alpha,
                             noise_beta=noise_beta, seed=seed, saliency_seqs=saliency_seqs,
                             saliency_files=list(saliency_files) if saliency_files else None,
                             student_model_name=student_model_name)
            for path in noise_record_paths:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(selection, f)
        print_acc(f"[serum_original.py] Targeted noise ({noise_target}): {selection['k']} of "
                  f"{selection['total_neurons']} MLP neurons, per layer {selection['per_layer_counts']}", print_message)

    if noise_alpha != 0.0:
        print_acc(f"[serum_original.py] Applying one-time shrink+perturb: noise={noise_alpha}", print_message)
        do_corruption(student_model, noise_alpha, noise_beta, neuron_masks=neuron_masks)



    # ----------------------------------------------------------------
    # Load distillation dataset
    # ----------------------------------------------------------------
    # Rank 0 builds the datasets cache first; the other ranks then read it instead of racing to write it.
    with accelerator.main_process_first():
        train_ds = load_train_dataset(train_files, tokenizer, max_length, join_or_subsequence, interleave_probs, seed,
                                      stopping_strategy, dataset_cache_dir, print_message)

    # Print size
    print_acc(f"[serum_original.py] Train dataset size: {len(train_ds)}", print_message)

    # Print 3 samples of distillation dataset text
    for i in range(3): 
        sample_ids = train_ds[i]["input_ids"]
        sample_text = tokenizer.decode(sample_ids, skip_special_tokens=False)
        print_acc(f'[serum_original.py] Sample distillation dataset text {i}: "{sample_text[:200]}..."', print_message)



    # ------------------------------------------------------------
    # Create DataLoaders
    # ------------------------------------------------------------
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="max_length", max_length=max_length)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  collate_fn=data_collator)

    # Determine steps
    steps_per_epoch = len(train_loader)
    effective_steps_per_epoch = steps_per_epoch // gradient_accumulation_steps
    total_steps = effective_steps_per_epoch * epochs
    if max_steps > 0:
        total_steps = min(total_steps, max_steps)
    if schedule_over_max_steps:
        assert max_steps > 0, "schedule_over_max_steps needs max_steps"
        total_steps = max_steps * accelerator.num_processes  # in ticks: num_processes per optimizer step
    if warmup_optimizer_steps is not None:
        assert schedule_over_max_steps, "warmup_optimizer_steps needs schedule_over_max_steps"
        assert 1 <= warmup_optimizer_steps < max_steps, (warmup_optimizer_steps, max_steps)
        num_warmup_steps = warmup_optimizer_steps * accelerator.num_processes  # in ticks, as total_steps
        print_acc(f"[serum_original.py] warmup: {warmup_optimizer_steps} optimizer steps = {num_warmup_steps} ticks",
                  print_message)
    print_acc(f"[serum_original.py] {steps_per_epoch} steps/epoch, total steps: {total_steps}", print_message)



    # ------------------------------------------------------------
    # Optimizer + Scheduler
    # ------------------------------------------------------------
    print_acc(f"[serum_original.py] Using AdamW, LR={learning_rate}, weight_decay={weight_decay}", print_message)
    optimizer = torch.optim.AdamW(student_model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda(scheduler_type, num_warmup_steps, total_steps, learning_rate, min_lr))



    # ------------------------------------------------------------
    # Accelerator prepare
    # ------------------------------------------------------------
    teacher_model, student_model, optimizer, train_loader, scheduler = accelerator.prepare(
        teacher_model, student_model, optimizer, train_loader, scheduler
    )

    # ------------------------------------------------------------
    # Get initial teacher eval dict which will be use for stop cond
    # ------------------------------------------------------------
    print_acc("[serum_original.py] Running evaluation ...", print_message)
    teacher_eval_dict = eval_fn(teacher_model, print_results=True) 
    teacher_eval_dict["train/step"] = 0
    teacher_eval_dict["train/tokens_seen"] = 0
    teacher_eval_dict["validation_stage"] = "teacher_baseline"

    if use_wandb and accelerator.is_main_process:
        wandb.log(teacher_eval_dict)
    if use_local_record and accelerator.is_main_process:
        with open(path_local_record, "a", encoding="utf-8") as f:
            f.write(json.dumps(teacher_eval_dict) + "\n")

    # ------------------------------------------------------------
    # Distillation Loop
    # ------------------------------------------------------------
    print_acc(f"[serum_original.py] Starting distillation ... (loss_type={loss_type}, domains={domain_names}, random_target_domain={random_target_domain})", print_message)
    global_step = 0
    global_tokens = 0
    teacher_model.eval()

    num_domains = len(domain_names)
    domain_loss_sum = torch.zeros(num_domains, device=accelerator.device)   # since the last optimizer step
    domain_token_count = torch.zeros(num_domains, device=accelerator.device)
    domain_tokens_seen = torch.zeros(num_domains, dtype=torch.float64)      # cumulative loss positions, all ranks
    target_generator = None
    target_probs = None
    if random_target_counts_path is not None:
        counts = torch.load(random_target_counts_path).double()
        target_probs = (counts / counts.sum()).float().to(accelerator.device)
        print_acc(f"[serum_original.py] Random targets drawn from {random_target_counts_path}", print_message)

    stop_early = False

    for epoch in range(epochs):
        print_acc(f"[serum_original.py] Epoch {epoch+1}/{epochs}", print_message)

        if shrink_perturb_repeat and noise_alpha != 0.0:
            print_acc(f"[serum_original.py] Re-applying shrink+perturb before epoch {epoch+1}", print_message)
            do_corruption(student_model, noise_alpha, noise_beta, neuron_masks=neuron_masks)

        student_model.train()
        micro_kd_sum = 0.0

        for step_in_epoch, batch in enumerate(train_loader):
            student_out = student_model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )

            if loss_type == "kl":
                with torch.no_grad():
                    teacher_out = teacher_model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        use_cache=False
                    )
                # Same mask and normalisation as upstream's forward_kl_loss_fn.
                labels = batch["input_ids"][..., 1:]
                token_mask = (labels != tokenizer.pad_token_id) & (labels != -100)
                token_loss = per_token_forward_kl(teacher_out.logits, student_out.logits)
            else:
                targets = batch["input_ids"]
                if random_target_domain is not None:
                    if target_generator is None:
                        target_generator = torch.Generator(device=targets.device)
                        target_generator.manual_seed(seed + 1000 * accelerator.process_index)
                    random_targets = sample_random_targets(
                        targets.shape, student_out.logits.size(-1), target_generator, targets.device, probs=target_probs
                    )
                    is_random_row = (batch["domain"] == domain_names.index(random_target_domain))[:, None]
                    targets = torch.where(is_random_row, random_targets, targets)
                # Mask by attention, not by target == pad id: a random target can be token 0.
                token_mask = batch["attention_mask"][..., 1:].bool()
                token_loss = per_token_cross_entropy(student_out.logits, targets)

            kd_loss = masked_mean(token_loss, token_mask)
            with torch.no_grad():
                row_domain = batch["domain"][:, None].expand_as(token_mask)
                for d in range(num_domains):
                    domain_mask = token_mask & (row_domain == d)
                    domain_loss_sum[d] += (token_loss.detach() * domain_mask).sum()
                    domain_token_count[d] += domain_mask.sum()

            micro_kd_sum += kd_loss.item()
            loss = kd_loss / gradient_accumulation_steps
            accelerator.backward(loss)

            tokens_this_batch = batch["attention_mask"].sum().detach()
            tokens_this_batch = accelerator.gather(tokens_this_batch).sum().item()
            global_tokens += tokens_this_batch

            if (step_in_epoch + 1) % gradient_accumulation_steps == 0:
                accelerator.clip_grad_norm_(student_model.parameters(), gradient_clipping_threshold)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                avg_kd_loss = micro_kd_sum / float(gradient_accumulation_steps)
                micro_kd_sum = 0.0
                # Per-domain mean loss over this optimizer step, pooled across ranks.
                domain_sums = accelerator.reduce(torch.cat([domain_loss_sum, domain_token_count]), reduction="sum")
                domain_loss_sum.zero_()
                domain_token_count.zero_()
                domain_tokens_seen += domain_sums[num_domains:].double().cpu()

                if global_step == 1 or global_step % 5 == 0:
                    print_acc(
                        f"[serum_original.py] Epoch {epoch+1}/{epochs}, {global_step}/{total_steps // accelerator.num_processes}, KD loss: {avg_kd_loss:.6f}",
                        print_message
                    )

                train_log_dict = {
                    "train/kd_loss" if loss_type == "kl" else "train/ce_loss": avg_kd_loss,
                    "train/step": global_step,
                    "train/tokens_seen": global_tokens,
                    "train/lr": scheduler.get_last_lr()[0],
                }
                for d, name in enumerate(domain_names):
                    count = domain_sums[num_domains + d].item()
                    if count > 0:
                        train_log_dict[f"train/loss_{name}"] = domain_sums[d].item() / count
                    train_log_dict[f"train/loss_tokens_{name}"] = domain_tokens_seen[d].item()
                if use_wandb and accelerator.is_main_process:
                    wandb.log(train_log_dict)
                if use_local_record and accelerator.is_main_process:
                    with open(path_local_record, "a", encoding="utf-8") as f:
                        f.write(json.dumps(train_log_dict) + "\n")

                if max_steps > 0 and global_step >= max_steps:
                    print_acc("[serum_original.py] Reached max_steps => Stopping.", print_message)
                    stop_early = True
                    break
                
                if (save_checkpoint_steps > 0 and (global_step % save_checkpoint_steps == 0)) or stop_early:
                    if accelerator.is_main_process:
                        checkpoint_name = "final_student" if stop_early else f"checkpoint-step{global_step}"
                        checkpoint_path = os.path.join(output_dir, checkpoint_name)
                        unwrapped_student = accelerator.unwrap_model(student_model)
                        unwrapped_student.save_pretrained(checkpoint_path)
                        tokenizer.save_pretrained(checkpoint_path)
                        print_acc(f"[serum_original.py] Saved checkpoint => {checkpoint_path}", print_message)  
                        
                # Validation: Calls provided eval function which returns a dictionary with eval name keys and results values
                # Do validation if it's the first stpe or the validation step or the last step
                if (global_step == 1 or global_step % validation_steps == 0 or stop_early):
                    print_acc("[serum_original.py] Running validation ...", print_message)
                    val_log_dict = eval_fn(student_model, print_results=True) 
                    val_log_dict["train/step"] = global_step
                    val_log_dict["train/tokens_seen"] = global_tokens
                    val_log_dict["validation_stage"] = "training_student"

                    if use_wandb and accelerator.is_main_process:
                        wandb.log(val_log_dict)
                    if use_local_record and accelerator.is_main_process:
                        with open(path_local_record, "a", encoding="utf-8") as f:
                            f.write(json.dumps(val_log_dict) + "\n")

                    should_stop = stop_cond_fn(student_eval_dict=val_log_dict, teacher_eval_dict=teacher_eval_dict)
                    # Convert to tensor and broadcast across all processes
                    should_stop_tensor = torch.tensor([1 if should_stop else 0], device=accelerator.device)
                    # All processes need to know if ANY process wants to stop
                    gathered_tensor = accelerator.gather(should_stop_tensor)
                    stop_early = gathered_tensor.sum().item() > 0

                    if stop_early:
                        print_acc(f"[serum_original.py] Early stopping triggered at step {global_step}", print_message)
                        break

        if stop_early:
            break

    # Add final validation after all training
    print_acc("[serum_original.py] Running final validation ...", print_message)
    final_val_dict = eval_fn(student_model, print_results=True)
    final_val_dict["train/step"] = global_step
    final_val_dict["train/tokens_seen"] = global_tokens
    final_val_dict["validation_stage"] = "final_student"

    if use_wandb and accelerator.is_main_process:
        wandb.log(final_val_dict)
    if use_local_record and accelerator.is_main_process:
        with open(path_local_record, "a", encoding="utf-8") as f:
            f.write(json.dumps(final_val_dict) + "\n")
            
    # Final model saving at the end of all epochs
    if accelerator.is_main_process:
        final_model_path = os.path.join(output_dir, "final_model")
        unwrapped_student = accelerator.unwrap_model(student_model)
        unwrapped_student.save_pretrained(final_model_path)
        tokenizer.save_pretrained(final_model_path)
        print_acc(f"[serum_original.py] Saved final model => {final_model_path}", print_message)


def load_tokenizer(name, cache_dir):
    """As partial_distill() loads the teacher's tokenizer (the packing cache fingerprint depends on its state)."""
    tokenizer = AutoTokenizer.from_pretrained(name, cache_dir=cache_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def load_train_dataset(train_files, tokenizer, max_length, join_or_subsequence, interleave_probs, seed,
                       stopping_strategy, dataset_cache_dir, print_message=True):
    """The packed, domain-tagged, interleaved train set (also builds its datasets cache; --warm_cache)."""
    train_ds_list = []
    for file in train_files:
        print_acc(f"[distill.py] Loading train dataset from {file}", print_message)
        train_ds = load_dataset("json", data_files=file, split="train", cache_dir=dataset_cache_dir)
        train_ds_list.append(train_ds)
    # ------------------------------------------------------------
    # PROCESS FOR SEQUENCE LENGTH
    # If join_or_subsequence, form sequences of exactly max_length 
    # by joining multiple or using subsequences
    # else filter for only those less than max_length
    # ------------------------------------------------------------
    train_ds_list, message = make_sequence_length(train_ds_list=train_ds_list, tokenizer=tokenizer, max_length=max_length, join_or_subsequence=join_or_subsequence)
    print_acc(message, print_message)
    # Tag each row with its source file, so the loop can pick targets and log loss per domain.
    train_ds_list = [ds.add_column("domain", [i] * len(ds)) for i, ds in enumerate(train_ds_list)]


    # Interleave dataset
    if len(train_ds_list) == 0:
        raise ValueError("No training dataset provided!")
    elif len(train_ds_list) == 1:
        train_ds = train_ds_list[0]
    else:
        print_acc(f"[serum_original.py] Interleaving with probabilities: {interleave_probs}", print_message)
        train_ds = interleave_datasets(train_ds_list, probabilities=interleave_probs, seed=seed, stopping_strategy=stopping_strategy)
    return train_ds


##############################################################
# The do_corruption function is used for "noise-and-decay"
##############################################################

def mlp_modules(model):
    """The (gate_proj, up_proj, down_proj) of each decoder layer of a Gemma(2)/Llama-style causal LM."""
    model = getattr(model, "module", model)
    return [(layer.mlp.gate_proj, layer.mlp.up_proj, layer.mlp.down_proj) for layer in model.model.layers]


@torch.no_grad()
def neuron_saliency(model, datasets, n_seqs, batch_size, tokenizer, max_length, device):
    """Mean |input to down_proj| per MLP neuron over the non-pad tokens of the first n_seqs rows of each dataset.
    Returns one [num_layers, intermediate_size] float64 tensor per dataset."""
    mlps = mlp_modules(model)
    collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="max_length", max_length=max_length)
    was_training = model.training
    model.eval()
    model.to(device)
    out = []
    for ds in datasets:
        assert len(ds) >= n_seqs, f"saliency needs {n_seqs} sequences, the dataset has {len(ds)}"
        sums = torch.zeros(len(mlps), mlps[0][2].in_features, dtype=torch.float64, device=device)
        count = torch.zeros((), dtype=torch.float64, device=device)
        mask = None

        def hook(i):
            def fn(module, args):
                sums[i] += (args[0].abs().double() * mask[..., None]).sum(dim=(0, 1))
            return fn
        handles = [down.register_forward_pre_hook(hook(i)) for i, (_, _, down) in enumerate(mlps)]
        try:
            for start in range(0, n_seqs, batch_size):
                rows = ds.select(range(start, min(start + batch_size, n_seqs)))
                batch = collator([{"input_ids": r["input_ids"], "attention_mask": r["attention_mask"]} for r in rows])
                mask = batch["attention_mask"].to(device).double()
                count += mask.sum()
                # The decoder only: the hooks need no logits (a 256k-vocab head would dominate memory).
                model.model(input_ids=batch["input_ids"].to(device), attention_mask=batch["attention_mask"].to(device))
        finally:
            for h in handles:
                h.remove()
        out.append((sums / count).cpu())
    model.train(was_training)
    return out


# Per-neuron stats: mean |h| and mean h (h = input to down_proj) and p_fire = fraction of tokens with
# act_fn(gate_proj x) > 0. Shared by run_neuron_stats_language.py and --noise_target neuron_rule.
NEURON_STATS = ("mean_abs", "mean", "p_fire")


@torch.no_grad()
def neuron_stats(model, ds, n_seqs, batch_size, tokenizer, max_length, device, ratio_seqs=64):
    """Per-neuron stats (run_neuron_stats_language.py; neuron_rule's p_fire) on the first n_seqs rows of ds, for splits 'all', 'h0', 'h1' and f'first{ratio_seqs}'.
    Returns {split: {stat: [num_layers, width] float64 ndarray}} and {split: n_tokens}."""
    assert len(ds) >= n_seqs >= 2, f"need {n_seqs} sequences, the dataset has {len(ds)}"
    half = n_seqs // 2
    mlps = mlp_modules(model)
    layers = getattr(model, "module", model).model.layers
    num_layers, width = len(mlps), mlps[0][2].in_features
    collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="max_length", max_length=max_length)
    was_training = model.training
    model.eval()
    model.to(device)
    # acc[split_index][stat]; split 0 = h0, 1 = h1, 2 = first ratio_seqs (a subset of h0/h1, accumulated alongside)
    acc = [{s: torch.zeros(num_layers, width, dtype=torch.float64, device=device) for s in NEURON_STATS} for _ in range(3)]
    tokens = [0.0, 0.0, 0.0]
    state = {}

    def down_hook(i):
        def fn(module, args):
            h, m = args[0].double(), state["mask"][..., None]
            for k in state["targets"]:
                acc[k]["mean_abs"][i] += (h.abs() * m).sum(dim=(0, 1))
                acc[k]["mean"][i] += (h * m).sum(dim=(0, 1))
        return fn

    def gate_hook(i):
        act_fn = layers[i].mlp.act_fn
        def fn(module, args, output):
            fire = (act_fn(output) > 0).double() * state["mask"][..., None]
            for k in state["targets"]:
                acc[k]["p_fire"][i] += fire.sum(dim=(0, 1))
        return fn

    handles = [down.register_forward_pre_hook(down_hook(i)) for i, (_, _, down) in enumerate(mlps)]
    handles += [gate.register_forward_hook(gate_hook(i)) for i, (gate, _, _) in enumerate(mlps)]
    try:
        # Batches never straddle a split boundary (half, ratio_seqs), so each batch has one set of targets.
        bounds = sorted({0, half, n_seqs, *( [ratio_seqs] if 0 < ratio_seqs < n_seqs else [] )})
        for lo, hi in zip(bounds[:-1], bounds[1:]):
            for start in range(lo, hi, batch_size):
                end = min(start + batch_size, hi)
                rows = ds.select(range(start, end))
                batch = collator([{"input_ids": r["input_ids"], "attention_mask": r["attention_mask"]} for r in rows])
                mask = batch["attention_mask"].to(device)
                state["mask"] = mask.double()
                state["targets"] = [0 if start < half else 1] + ([2] if start < ratio_seqs else [])
                n = mask.sum().item()
                for k in state["targets"]:
                    tokens[k] += n
                model.model(input_ids=batch["input_ids"].to(device), attention_mask=mask)
    finally:
        for h in handles:
            h.remove()
    out, n_tokens = {}, {}
    names = ["h0", "h1", f"first{ratio_seqs}"]
    model.train(was_training)
    for k, name in enumerate(names):
        if tokens[k]:
            out[name] = {s: (acc[k][s] / tokens[k]).cpu().numpy() for s in NEURON_STATS}
            n_tokens[name] = int(tokens[k])
    tot = tokens[0] + tokens[1]
    out["all"] = {s: ((acc[0][s] + acc[1][s]) / tot).cpu().numpy() for s in NEURON_STATS}
    n_tokens["all"] = int(tot)
    return out, n_tokens


def select_mlp_neurons(model, noise_target, noise_frac, seed, accelerator=None, saliency_ds=None, saliency_seqs=64,
                       batch_size=4, tokenizer=None, max_length=None, eps=1e-6, noise_count=None, p_kor_min=None,
                       p_eng_max=None, device=None):
    """Pick MLP neurons across all layers to noise.
      neuron_act   k = round(noise_frac x all MLP neurons): global top-k of mean|act| on forget (saliency_ds[0]) /
                   (mean|act| on retain (saliency_ds[1]) + eps)
      neuron_rand  k = round(noise_frac x all) or exactly noise_count (one of the two), uniformly at random, from a
                   torch.Generator seeded with `seed` (the global RNG is not used)
      neuron_rule  every neuron with p_fire(forget) >= p_kor_min and p_fire(retain) <= p_eng_max, p_fire as
                   run_neuron_stats_language.py computes it (neuron_stats, split 'all') on saliency_seqs sequences each;
                   k is however many pass
    Scores are computed on `device` (default: the accelerator's, else CPU); with several processes rank 0's are
    broadcast, so every rank selects the same neurons.
    Returns ([num_layers] bool masks over each layer's neurons, a JSON-able record of the selection)."""
    mlps = mlp_modules(model)
    num_layers, width = len(mlps), mlps[0][2].in_features
    total = num_layers * width
    if device is None:
        device = accelerator.device if accelerator is not None else torch.device("cpu")
    multi = accelerator is not None and accelerator.num_processes > 1
    if noise_target == "neuron_rule":
        assert noise_frac is None and noise_count is None, "neuron_rule selects by threshold, not by size"
        assert p_kor_min is not None and p_eng_max is not None, "neuron_rule needs p_kor_min and p_eng_max"
        k = None
    elif noise_target == "neuron_rand" and noise_count is not None:
        assert noise_frac is None, "neuron_rand takes noise_frac or noise_count, not both"
        k = int(noise_count)
    else:
        assert noise_count is None, "noise_count applies only to neuron_rand"
        k = round(noise_frac * total)
    assert k is None or 0 < k <= total, (noise_frac, noise_count, total)
    record = {"num_layers": num_layers, "intermediate_size": width, "total_neurons": total, "k": k}
    if noise_target == "neuron_rule":
        p_fire, n_tokens = [], []
        for ds in saliency_ds:
            stats, n_tok = neuron_stats(model, ds, saliency_seqs, batch_size, tokenizer, max_length, device,
                                        ratio_seqs=0)
            p_fire.append(torch.from_numpy(stats["all"]["p_fire"]))
            n_tokens.append(n_tok["all"])
        p = torch.stack(p_fire)   # [2, num_layers, width] float64: forget, retain
        if multi:
            from accelerate.utils import broadcast
            p = broadcast(p.to(accelerator.device), from_process=0).cpu()
        p_kor, p_eng = p[0].flatten(), p[1].flatten()
        chosen = torch.nonzero((p_kor >= p_kor_min) & (p_eng <= p_eng_max)).flatten()
        k = record["k"] = len(chosen)
        assert k > 0, f"no neuron has p_fire(kor) >= {p_kor_min} and p_fire(eng) <= {p_eng_max}"
        record.update(p_kor_min=p_kor_min, p_eng_max=p_eng_max, n_tokens_kor=n_tokens[0], n_tokens_eng=n_tokens[1],
                      p_fire_kor_mean_per_layer=p[0].mean(dim=1).tolist(),
                      p_fire_eng_mean_per_layer=p[1].mean(dim=1).tolist())
    elif noise_target == "neuron_rand":
        generator = torch.Generator().manual_seed(seed)
        chosen = torch.randperm(total, generator=generator)[:k]
    elif noise_target == "neuron_act":
        kor, eng = neuron_saliency(model, saliency_ds, saliency_seqs, batch_size, tokenizer, max_length, device)
        scores = (kor / (eng + eps)).flatten()
        if multi:
            from accelerate.utils import broadcast
            scores = broadcast(scores.to(accelerator.device), from_process=0).cpu()
        chosen = torch.topk(scores, k).indices
        top, median = scores[chosen].min().item(), scores.median().item()
        record["score_summary"] = {
            "eps": eps, "max": scores.max().item(), "kth_selected": top, "median_all": median,
            "kth_selected_over_median": top / median, "median_selected": scores[chosen].median().item(),
            "min": scores.min().item(),
            "selected_median_over_median_all": scores[chosen].median().item() / median}
        record["mean_abs_act_kor_per_layer"] = kor.mean(dim=1).tolist()
        record["mean_abs_act_eng_per_layer"] = eng.mean(dim=1).tolist()
    else:
        raise ValueError(noise_target)
    flat = torch.zeros(total, dtype=torch.bool)
    flat[chosen] = True
    masks = list(flat.view(num_layers, width))
    record["per_layer_counts"] = [int(m.sum()) for m in masks]
    record["selected"] = [torch.nonzero(m).flatten().tolist() for m in masks]
    if noise_target == "neuron_rule":   # aligned with record["selected"]
        record["p_kor_selected"] = [p[0][i][m].tolist() for i, m in enumerate(masks)]
        record["p_eng_selected"] = [p[1][i][m].tolist() for i, m in enumerate(masks)]
    return masks, record


def do_corruption(model, noise_alpha, noise_beta = 0.1, seed = 42, neuron_masks=None):
    # Loop through all parameters and add random noise scaled by scale factor
    assert 0 <= noise_alpha <= 1
    assert 0 <= noise_beta <= 1

    if neuron_masks is not None:
        return do_targeted_corruption(model, noise_alpha, noise_beta, neuron_masks)

    for param in model.parameters():
        if param.requires_grad:
            corruption = torch.zeros_like(param.data)
            if len(param.data.shape) == 2:
                noise = torch.nn.init.xavier_uniform_(
                    torch.empty_like(param.data)
                )
            elif len(param.data.shape) == 1:
                noise = torch.zeros_like(param.data)
            else:
                raise RuntimeError(
                    f"Unsupported parameter shape: {param.data.shape}"
                )
            corruption = noise_beta * noise

            param.data = (1 - noise_alpha) * param.data + noise_alpha * corruption

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

@torch.no_grad()
def do_targeted_corruption(model, noise_alpha, noise_beta, neuron_masks):
    """AGI-2635: theta <- (1 - alpha) theta + alpha beta N on the selected MLP neurons only (gate_proj and up_proj
    rows, down_proj columns); every other parameter is untouched. N is drawn exactly as do_corruption draws it: one
    full-shape xavier_uniform tensor per 2-D parameter, in model.parameters() order, so the fan-in/fan-out are the
    full matrix's and the global RNG ends where untargeted noise leaves it (same data order downstream)."""
    targets = {}
    for i, (gate, up, down) in enumerate(mlp_modules(model)):
        rows = neuron_masks[i].to(gate.weight.device)
        targets[id(gate.weight)] = (rows, 0)
        targets[id(up.weight)] = (rows, 0)
        targets[id(down.weight)] = (rows.to(down.weight.device), 1)
    hit = 0
    for param in model.parameters():
        if not param.requires_grad:
            continue
        if len(param.data.shape) == 2:
            noise = torch.nn.init.xavier_uniform_(torch.empty_like(param.data))
        elif len(param.data.shape) == 1:
            noise = None
        else:
            raise RuntimeError(f"Unsupported parameter shape: {param.data.shape}")
        if id(param) not in targets:
            continue
        sel, dim = targets[id(param)]
        hit += 1
        if dim == 0:
            param.data[sel] = (1 - noise_alpha) * param.data[sel] + noise_alpha * (noise_beta * noise[sel])
        else:
            param.data[:, sel] = (1 - noise_alpha) * param.data[:, sel] + noise_alpha * (noise_beta * noise[:, sel])
    assert hit == 3 * len(neuron_masks), f"noised {hit} MLP matrices, expected {3 * len(neuron_masks)}"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
