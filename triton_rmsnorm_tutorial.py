import torch
import triton
import triton.language as tl

# =====================================================================
# 1. THE TRITON KERNEL (This runs ON THE GPU)
# =====================================================================
# The @triton.jit decorator tells the Triton compiler to turn this Python
# function into highly optimized PTX (Nvidia machine code).
@triton.jit
def rmsnorm_kernel(
    # Pointers to memory locations in VRAM
    x_ptr,      # Input tensor pointer
    weight_ptr, # RMSNorm learnable weights pointer
    out_ptr,    # Output tensor pointer

    # Meta-parameters
    stride_x_row, # How many memory addresses to jump to get to the next row in X
    n_cols,       # The embedding dimension (d_model), e.g., 768
    eps,          # Epsilon to prevent division by zero

    # BLOCK_SIZE is a compile-time constant. It defines how many elements
    # one block of threads handles. We set it to the next power of 2 >= n_cols.
    BLOCK_SIZE: tl.constexpr,
):
    # 1. Identify which row this specific thread block is working on.
    # We assign one thread block per row (token) in our batch.
    row_idx = tl.program_id(0)

    # 2. Calculate the memory address where this row starts in the input tensor.
    row_start_ptr = x_ptr + row_idx * stride_x_row
    out_row_start_ptr = out_ptr + row_idx * stride_x_row

    # 3. Create a vector of offsets: [0, 1, 2, ..., BLOCK_SIZE-1]
    offsets = tl.arange(0, BLOCK_SIZE)

    # 4. Create a mask. If BLOCK_SIZE is 1024, but our d_model (n_cols) is 768,
    # we need to mask out threads 768-1023 so they don't do invalid memory reads.
    mask = offsets < n_cols

    # 5. LOAD DATA FROM VRAM TO SRAM (Registers)
    # This is the magic. We load the entire row and the weights at once.
    x = tl.load(row_start_ptr + offsets, mask=mask, other=0.0)
    weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0)

    # 6. DO THE MATH IN SRAM (Super fast!)
    # RMSNorm formula: x / sqrt(mean(x^2) + eps) * weight

    # a. Square every element
    x_sq = x * x

    # b. Sum the squares across the row, then divide by n_cols to get the mean.
    # Note: tl.sum works across the BLOCK_SIZE dimension.
    mean_sq = tl.sum(x_sq, axis=0) / n_cols

    # c. Calculate the inverse square root (rsqrt is a fast hardware instruction)
    rsqrt = tl.math.rsqrt(mean_sq + eps)

    # d. Normalize and scale by the weight
    output = x * rsqrt * weight

    # 7. STORE THE RESULT BACK TO VRAM
    tl.store(out_row_start_ptr + offsets, output, mask=mask)


# =====================================================================
# 2. THE PYTHON WRAPPER (This runs ON THE CPU)
# =====================================================================
# This function prepares the tensors and launches the kernel.
def triton_rmsnorm(x, weight, eps=1e-5):
    # Ensure inputs are contiguous in memory
    x = x.contiguous()
    weight = weight.contiguous()

    # We want output to have the same shape as input
    out = torch.empty_like(x)

    # We flatten the batch and sequence dimensions into 'rows'.
    # Example: shape (batch=2, seq=512, d_model=768) -> (1024 rows, 768 cols)
    x_2d = x.view(-1, x.shape[-1])
    n_rows, n_cols = x_2d.shape

    # Find the next power of 2 greater than n_cols for our block size.
    # Triton requires BLOCK_SIZE to be a power of 2.
    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    # grid defines how many thread blocks to launch.
    # We launch 1 block per row.
    grid = (n_rows,)

    # Launch the kernel!
    rmsnorm_kernel[grid](
        x, weight, out,       # Pointers
        x_2d.stride(0),       # Stride for jumping rows
        n_cols, eps,          # Metas
        BLOCK_SIZE=BLOCK_SIZE # Constants
    )

    return out

# =====================================================================
# 3. TEST AND COMPARE
# =====================================================================
if __name__ == "__main__":
    print("Testing Triton RMSNorm vs PyTorch RMSNorm...")
    torch.manual_seed(0)

    # Create fake data
    batch, seq_len, d_model = 2, 512, 768
    x = torch.randn(batch, seq_len, d_model, device='cuda')
    weight = torch.ones(d_model, device='cuda')

    # Run Triton
    out_triton = triton_rmsnorm(x, weight)

    # Run Standard PyTorch (like in your model.py)
    norm = x.pow(2).mean(-1, keepdim=True)
    x_normed = x * torch.rsqrt(norm + 1e-5)
    out_pytorch = weight * x_normed

    # Check if they match!
    max_diff = torch.max(torch.abs(out_triton - out_pytorch))
    print(f"Maximum difference between Triton and PyTorch: {max_diff.item():.6f}")
    if max_diff < 1e-5:
        print("✅ SUCCESS! The Triton kernel works perfectly.")
    else:
        print("❌ FAILED! Outputs do not match.")
