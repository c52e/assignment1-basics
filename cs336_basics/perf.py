def CalcMemory(batch_size, vocab_size, context_length, num_layers, d_model, num_heads):
    d_ff = 4 * d_model
    d_k = d_model // num_heads * num_heads
    def CalcModelParamsCount():
        rms_norm = d_model * (2 * num_layers + 1)
        atten = d_model * d_k * 4 * num_layers
        feedforward = d_model * d_ff * 3 * num_layers
        output_embedding = d_model * vocab_size
        return rms_norm + atten + feedforward + output_embedding
    model_params_count = CalcModelParamsCount()

    memory_model_params = 4 * model_params_count / 1024**3
    memory_gradients = 4 * model_params_count / 1024**3
    memory_adamw = 8 * model_params_count / 1024**3
    
    def CalcActivationParamsCount():
        def TransformerBlock():
            rms_norm = batch_size * context_length * d_model * 2 * 2

            qkv = batch_size * context_length * d_model
            qkv_out = batch_size * context_length * d_k * 3
            softmax = batch_size * context_length * context_length * 3 * num_heads
            outproj = batch_size * context_length * d_k

            w1 = batch_size * context_length * d_model
            silu = batch_size * context_length * d_model * 2
            w2 = batch_size * context_length * d_model

            return rms_norm + qkv + qkv_out + softmax + outproj + w1 + silu + w2
    
        transformer_block = TransformerBlock() * num_layers
        final_rms_norm = batch_size * context_length * d_model
        output_embedding = batch_size * context_length * d_model
        cross_entropy = batch_size * context_length * vocab_size
        return transformer_block + final_rms_norm + output_embedding + cross_entropy
    memory_activation = 4 * CalcActivationParamsCount() / 1024**3
    memory_sum = memory_model_params + memory_gradients + memory_adamw + memory_activation
    return memory_model_params, memory_gradients, memory_adamw, memory_activation, memory_sum

def CalcFlops(batch_size, vocab_size, context_length, num_layers, d_model, num_heads):
    d_ff = 4 * d_model
    d_k = d_model // num_heads * num_heads
    op_mha = 2 * d_model * d_k * context_length * 4 + 2 * context_length * context_length * d_k * 2
    op_swiglu = 2 * d_model * d_ff * context_length * 3
    op_per_layer = op_mha + op_swiglu
    op_layers = op_per_layer * num_layers
    op_lm_head = 2 * d_model * vocab_size * context_length
    op_sum = op_layers + op_lm_head
    return op_sum * 3 * batch_size
