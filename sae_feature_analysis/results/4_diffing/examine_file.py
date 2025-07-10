import torch

BASE_FILE = '/workspace/results/4_diffing/1_llama_trainer32x_layer15_base.pt'
print('Loading file...')
data = torch.load(BASE_FILE)
print('Top-level keys:', list(data.keys()))
print('Metadata:', data['metadata'])
print()

for key in data.keys():
    if key != 'metadata':
        print('Token type', key, 'contains:')
        token_data = data[key]
        if isinstance(token_data, dict):
            for subkey in token_data.keys():
                tensor = token_data[subkey]
                print('  ', subkey, ': shape', tensor.shape, ', dtype', tensor.dtype)
        else:
            print('  Direct tensor: shape', token_data.shape, ', dtype', token_data.dtype)
        print()