import torch
import time
import numpy as np
import matplotlib.pyplot as plt

def test_batch_sizes(state_size=14, action_size=3, iterations=50):
    """
    Test performance of different batch sizes on available GPU
    
    Parameters:
    - state_size: Dimension of the game state
    - action_size: Number of possible actions
    - iterations: Number of iterations to average performance
    """
    # Batch sizes to test
    batch_sizes = [16, 32, 64, 128, 256]
    
    # Check CUDA availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on: {device}")
    
    # Performance metrics
    memory_usage = []
    training_times = []
    
    # Create placeholder network similar to your neural network
    class PlaceholderNetwork(torch.nn.Module):
        def __init__(self, state_size, action_size):
            super().__init__()
            self.layers = torch.nn.Sequential(
                torch.nn.Linear(state_size, 64),
                torch.nn.ReLU(),
                torch.nn.Linear(64, 32),
                torch.nn.ReLU(),
                torch.nn.Linear(32, action_size)
            )
        
        def forward(self, x):
            return self.layers(x)
    
    # Test each batch size
    for batch_size in batch_sizes:
        # Reset memory cache
        torch.cuda.empty_cache()
        
        # Create model and optimizer
        model = PlaceholderNetwork(state_size, action_size).to(device)
        optimizer = torch.optim.Adam(model.parameters())
        criterion = torch.nn.MSELoss()
        
        # Create sample data
        states = torch.rand(batch_size, state_size, device=device)
        targets = torch.rand(batch_size, action_size, device=device)
        
        # Warm-up iteration
        model(states)
        
        # Track performance
        batch_memory_usage = []
        batch_training_times = []
        
        # Run iterations
        for _ in range(iterations):
            # Start timing and memory tracking
            torch.cuda.reset_peak_memory_stats()
            start_time = time.time()
            
            # Training step
            optimizer.zero_grad()
            outputs = model(states)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            # End timing
            end_time = time.time()
            
            # Memory usage
            current_memory = torch.cuda.max_memory_allocated(device) / 1024**2  # MB
            batch_memory_usage.append(current_memory)
            batch_training_times.append(end_time - start_time)
        
        # Store average metrics
        memory_usage.append(np.mean(batch_memory_usage))
        training_times.append(np.mean(batch_training_times) * 1000)  # Convert to milliseconds
    
    # Visualization
    plt.figure(figsize=(12, 5))
    
    # Memory Usage subplot
    plt.subplot(1, 2, 1)
    plt.plot(batch_sizes, memory_usage, marker='o')
    plt.title('GPU Memory Usage')
    plt.xlabel('Batch Size')
    plt.ylabel('Memory (MB)')
    plt.xscale('log')
    
    # Training Time subplot
    plt.subplot(1, 2, 2)
    plt.plot(batch_sizes, training_times, marker='o')
    plt.title('Training Time')
    plt.xlabel('Batch Size')
    plt.ylabel('Time (ms)')
    plt.xscale('log')
    
    plt.tight_layout()
    plt.savefig('batch_size_performance.png')
    plt.close()
    
    # Print results
    print("\nBatch Size Performance Analysis:")
    print("Batch Size | Avg Memory (MB) | Avg Training Time (ms)")
    print("-" * 50)
    for bs, mem, time_ms in zip(batch_sizes, memory_usage, training_times):
        print(f"{bs:10d} | {mem:14.2f} | {time_ms:18.2f}")

# Run the analysis
test_batch_sizes()