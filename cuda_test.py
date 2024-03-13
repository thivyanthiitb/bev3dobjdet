import torch
import debugpy
debugpy.listen(5690)
print("Waiting for debugger attach")
debugpy.wait_for_client()
print("Debugger attached")
# Check if CUDA is available
if torch.cuda.is_available():
    print("CUDA is available. Here are the CUDA devices:")
    # List available CUDA devices
    for i in range(torch.cuda.device_count()):
        print(f"Device {i}: {torch.cuda.get_device_name(i)}")
    
    # Select a device
    device = torch.device("cuda:0")  # Assuming you're using the first CUDA device

    # Create two tensors and move them to GPU
    x = torch.tensor([1.0, 2.0, 3.0], device=device)
    y = torch.tensor([4.0, 5.0, 6.0], device=device)

    # Perform a simple addition operation
    z = x + y
    print("Result of the tensor addition: ", z)

    # Check if the operation was performed on the GPU
    print("Is the computation on GPU?", z.is_cuda)
else:
    print("CUDA is not available. Please check your PyTorch installation and GPU drivers.")
