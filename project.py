import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

# Load the model
model = torch.load('model.pth')

# Prune the model
prune.random_unstructured(model, 'weight', amount=0.2)

# Run the optimized model
input_image = ...
output = model(input_image)

# Print the output
print(output)

# Measure the inference time
import time
start_time = time.time()
output = model(input_image)
end_time = time.time()
print(f'Inference time: {end_time - start_time} seconds')





# Original model
original_model = torch.load('original_model.pth')
original_output = original_model(input_image)
original_inference_time = ...

# Optimized model
optimized_model = torch.load('optimized_model.pth')
optimized_output = optimized_model(input_image)
optimized_inference_time = ...
# Print the comparison
print(f'Original inference time: {original_inference_time} seconds')
print(f'Optimized inference time: {optimized_inference_time} seconds')
print(f'Original output: {original_output}')
print(f'Optimized output: {optimized_output}')
