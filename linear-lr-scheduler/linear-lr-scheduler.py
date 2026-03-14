def linear_lr(step, total_steps, initial_lr, final_lr=0.0, warmup_steps=0) -> float:
    """
    Linear warmup (0→initial_lr) then linear decay (initial_lr→final_lr).
    Steps are 0-based; clamp at final_lr after total_steps.
    """
    # 1. Beyond total steps: stay at final_lr
    if step >= total_steps:
        return float(final_lr)
    
    # 2. Warmup phase: 0 -> initial_lr
    if step < warmup_steps:
        return (step / warmup_steps) * initial_lr
    
    # 3. Decay phase: initial_lr -> final_lr
    # Linear interpolation between η0 and ηf based on progress from W to T
    decay_steps = total_steps - warmup_steps
    if decay_steps <= 0:
        return float(final_lr)
        
    current_decay_step = step - warmup_steps
    decay_ratio = current_decay_step / decay_steps
    
    return initial_lr - (decay_ratio * (initial_lr - final_lr))
