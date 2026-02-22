""" 
Predictive Coding Network (PCN) implementation in PyTorch.

Directly inspired by Stenlund, M., 2025. Introduction to Predictive Coding Networks for Machine Learning. arXiv preprint arXiv:2506.06332.

Please checkout their repository for the original code and more details: https://github.com/Monadillo/pcn-intro

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.amp import autocast
import plotly.graph_objects as go
from plotly import colors
import numpy as np
import matplotlib.pyplot as plt
from statistics import mean

def plot_energy_history_interactive(energy_history, T_infer, T_learn,
                                    title='Batch-averaged Energy Trajectories'
                                    ):
    """
    Interactive energy trajectories with Plotly.
    - energy_history: list of epochs; each epoch is list of batch energy lists.
    - T_infer: number of inference steps.
    - T_learn: number of learning steps.
    Hover to see Epoch, Batch, Step, Energy.
    """
    num_epochs = len(energy_history)
    # Sample one color per epoch from Viridis
    epoch_colors = colors.sample_colorscale(
        colors.sequential.Viridis,
        [i/(num_epochs-1) if num_epochs>1 else 0 for i in range(num_epochs)]
    )

    fig = go.Figure()
    for epoch_idx, epoch_energies in enumerate(energy_history):
        color = epoch_colors[epoch_idx]
        for batch_idx, batch_vals in enumerate(epoch_energies):
            steps = list(range(len(batch_vals)))
            # Attach epoch and batch metadata for hover
            customdata = [[epoch_idx+1, batch_idx+1] for _ in steps]
            fig.add_trace(go.Scatter(
                x=steps,
                y=batch_vals,
                mode='lines',
                line=dict(color=color, width=1),
                hovertemplate=(
                    'Epoch %{customdata[0]}<br>'
                    'Batch %{customdata[1]}<br>'
                    'Step %{x}<br>'
                    'Energy %{y:.4f}<extra></extra>'
                ),
                customdata=customdata,
                showlegend=False
            ))
    # Vertical separator between inference and learning
    sep_position = T_infer + 0.5
    fig.add_vline(
        x=sep_position,
        line_dash='dash',
        line_color='black',
        annotation_text='Inference end',
        annotation_position='top right'
    )

    fig.update_layout(
        title=title,
        xaxis_title='Step t (Inference then Learning)',
        yaxis_title='Energy'
    )

    # Optional: log-scale y-axis if needed
    # fig.update_yaxes(type='log')

    fig.show()
    
def plot_epoch_avg_interactive(energy_history, T_infer, T_learn,
                               title='Batch-Averaged Energy Trajectories (Mean ± 1-std)'
                              ):
    """
    Interactive per-epoch mean ±1-std energy trajectories using Plotly.
    - energy_history: list of epochs; each epoch is list of batch energy trajectories.
    - T_infer: number of inference steps.
    - T_learn: number of learning steps.
    Hover to see Epoch, Step, Mean Energy.
    """
    num_epochs = len(energy_history)
    epoch_colors = colors.sample_colorscale(
        colors.sequential.Viridis,
        [i/(num_epochs-1) if num_epochs>1 else 0 for i in range(num_epochs)],
        colortype='hex'   # Makes each color something like '#440154'
    )

    epoch_colors = [
        c if isinstance(c, str)
          else '#{0:02x}{1:02x}{2:02x}'.format(
              *(int(round(255*v)) for v in c)
          )
        for c in epoch_colors
    ]

    fig = go.Figure()
    total_steps = T_infer + T_learn

    for epoch_idx, epoch_batches in enumerate(energy_history):
        arr = np.array(epoch_batches)  # shape (n_batches, total_steps+1)
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        steps = list(range(len(mean)))
        color = epoch_colors[epoch_idx]
        # Upper bound (mean + std)
        fig.add_trace(go.Scatter(
            x=steps,
            y=mean + std,
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        ))

        # Lower bound (mean - std) with fill
        fig.add_trace(go.Scatter(
            x=steps,
            y=mean - std,
            mode='lines',
            fill='tonexty',
            # Convert hex '#rrggbb' → rgba(r,g,b,0.2)
            fillcolor=f"rgba({int(color[1:3],16)},"
                      f"{int(color[3:5],16)},"
                      f"{int(color[5:7],16)},0.2)",
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip'
        ))

        # Mean line
        fig.add_trace(go.Scatter(
            x=steps,
            y=mean,
            mode='lines',
            line=dict(color=color, width=2),
            name=f'Epoch {epoch_idx+1}',
            hovertemplate=(
                f'Epoch {epoch_idx+1}<br>'
                'Step %{x}<br>'
                'Energy %{y:.4f}<extra></extra>'
            )
        ))


    # Vertical separator between inference and learning
    fig.add_vline(
        x=T_infer + 0.5,
        line_dash='dash',
        line_color='black',
        annotation_text='Inference end',
        annotation_position='top right'
    )

    fig.update_layout(
        title=title,
        xaxis_title='Step t (Inference then Learning)',
        yaxis_title='Energy'
    )

    # Optional: log-scale y-axis if needed
    # fig.update_yaxes(type='log')

    fig.show()


# Define latent layer
class PCNLayer(nn.Module):
    def __init__(self,
                 in_dim,   # d_{l+1}  - dimension of layer above
                 out_dim,  # d_l      - dimension of current layer
                 device,
                 activation_fn=torch.relu, # nonlinearity f^(l)
                 activation_deriv=lambda a: (a > 0).float() # derivative f^(l)',
                 ):
        
        super().__init__()
        self.W = nn.Parameter(torch.empty(out_dim, in_dim)) # W^(l)
        # self.K = torch.eye(out_dim).to(device)
        self.K = self.gaussian_kernel(out_dim, c_sigma = 0.1, e_sigma=0.2, e_weight = 0.1).to(device) # Gaussian Kernel K for receptive field
        
        print(self.K.shape)
        nn.init.xavier_uniform_(self.W)
        self.activation_fn     = activation_fn
        self.activation_deriv  = activation_deriv

    def forward(self, x_above):
        with autocast(device_type='cuda'):
            a     = x_above @ self.W.T          # A^(l) = X^(l+1) @ {W^(l)}^T
            f_a = self.activation_fn(a)         # \hat X^(l) = f^(l)(K^(l) @ A^(l))
            x_hat   = f_a @ self.K.to(a.dtype)  # K is a symmetric matrix
            
            return x_hat, a
    
    # Composite Gaussian Kernel
    def gaussian_kernel(self, n = 10, c_sigma=2, e_sigma=16, e_weight = 0.5):
        K = np.zeros((n,n))
        for idx in range(n): 
            C_idx = np.exp(-((idx-np.arange(n))**2)*(1/c_sigma)) 
            E_idx = e_weight * np.exp(-((idx-np.arange(n))**2)*(1/e_sigma)) 
            K [idx,:] = C_idx - E_idx
        return torch.tensor(K)

# Define network structure
class PredictiveCodingNetwork(nn.Module):
    def __init__(self,
                 dims,        # [d_0,...,d_L]  - list of layer dimensions
                 output_dim,   # d_out          - readout layer dimension
                 device
                 ):
        super().__init__()
        self.dims = dims
        self.L = len(dims) - 1            # L  - number of latent layers
        self.layers = nn.ModuleList([     # Build latent layers
            PCNLayer(in_dim=dims[l+1],    # Layer l reads from layer l+1
                     out_dim=dims[l],
                     device=device)
            for l in range(self.L)        # l = 0,...,L-1
        ])
        # Build readout layer: maps top latent X^(L) to
        # predicted output \hat Y
        # Note: nn.Linear applies (batch, in_features) @ weight.T under the hood,
        # which corresponds exactly to X^(L) @ (W^out)^T
        self.readout = nn.Linear(dims[-1], output_dim, bias=False)


    def init_latents(self, batch_size, device):
        # returns [X^(1),...,X^(L)] as random normals
        return [
           torch.randn(batch_size, d, device=device, requires_grad=False)
           for d in self.dims[1:]
        ]

    def compute_errors(self, inputs_latents):
        # Compute predictions from input and latent variables
        # Argument: inputs_latents - list of tensors [X^(0), X^(1),...,X^(L)] shaped [(B,d_0),...,(B,d_L)]
        # Returns: two lists of tensors shaped [(B,d_0),...,(B,d_{L-1})]
        errors, gain_modulated_errors = [], []
        for l, layer in enumerate(self.layers):       # l = 0,...,L-1
            # Call to layer returns:
            #   a = X^(l+1) @ W^(l).T  (preactivations A^(l))
            #   K_a = K^(l) @ a        (Gaussian Kernel Applied K^(l))
            #   x_hat = f^(l)(K_a)       (predictions \hat X^(l))
            x_hat, a  = layer(inputs_latents[l + 1])
            err       = inputs_latents[l] - x_hat
            a_activation_deriv = layer.activation_deriv(a) 
            a_activation = (a_activation_deriv @ layer.K.to(a_activation_deriv.dtype))
            gm_err    = err * a_activation
            errors.append(err)                               # E^(l) - prediction errors
            gain_modulated_errors.append(gm_err)             # H^(l) - gain-modulated errors
        return errors, gain_modulated_errors
    


def train_pcn(model, data_loader, num_epochs, eta_infer, eta_learn, T_infer, T_learn, device='cuda'):
    model.to(device)
    model.train()

    energy_history = []             # Will hold per-epoch batch-averaged energy trajectories
    supervised_energy_history = []  # Will hold per-epoch batch-averaged supervised energy trajectories

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1} / {num_epochs}")

        epoch_energies = []             # Will hold batch-averaged energy trajectories for this epoch
        epoch_supervised_energies = []  # Will hold batch-averaged supervised energy trajectories for this epoch

        progress_bar = tqdm(data_loader)
        for x_batch, y_batch in progress_bar:
            B = x_batch.size(0)
            d_0 = model.dims[0]
            batch_energies = []            # Will hold the batch-averaged energy trajectory for this batch
            batch_supervised_energies = [] # Will hold the batch-averaged supervised energy trajectory for this batch

            # Flatten inputs
            x_batch = x_batch.view(B, d_0).to(device)
            # Convert ouputs to one-hot vectors
            y_batch = F.one_hot(y_batch, num_classes=model.readout.out_features).float().to(device)
            # Concatenate inputs X^(0) and initialized latents X^(l) - shape [(B,d_0),...,(B,d_L)]
            inputs_latents = [x_batch] + model.init_latents(B, device)
            # Weights W^(0),...,W^(L-1),W^out of latent and output layers - shape [(d_0,d_1),...,(d_{L-1},d_L),(d_L,d_out)]
            weights = [layer.W for layer in model.layers] + [model.readout.weight]

            # Prediction block - for energy at step t=0 and inference at step t=1
            # Compute E^(l) and H^(l) for l=0,...,L-1
            errors, gain_modulated_errors = model.compute_errors(inputs_latents)
            # Compute \hat Y from X^(L)
            y_hat           = model.readout(inputs_latents[-1]) # X^(L) @ W^out.T
            # Compute E^sup
            eps_sup         = y_hat - y_batch
            # Compute E^(L)
            eps_L           = eps_sup @ weights[-1]
            # Append to E^(L) to errors [E^(0),...,E^(L-1)]
            errors_extended = errors + [eps_L]


            # Record initial batch-averaged energy before any updates (t=0)
            supervised_energy = 0.5 * eps_sup.pow(2).sum().item() / B
            latent_energy     = 0.5 * sum(e.pow(2).sum().item() for e in errors) / B
            batch_supervised_energies.append(supervised_energy)
            batch_energies.append(latent_energy + supervised_energy)


            # INFERENCE - T_infer steps
            with torch.no_grad(), autocast(device_type='cuda'):
                for t in range(1, T_infer + 1):
                    
                    # Predictions for this inference step t have already been computed
                    # Inference updates - Gradient step for latents X^(1),...,X^(L)
                    for l in range(1, model.L + 1):  # l=1,...,L
                        grad_Xl = errors_extended[l] - gain_modulated_errors[l-1] @  weights[l-1] 
                        inputs_latents[l] -= eta_infer * grad_Xl

                    # Prediction block - for energies at the end of this step t, and for inference/learning at step t+1
                    errors, gain_modulated_errors = model.compute_errors(inputs_latents)
                    y_hat           = model.readout(inputs_latents[-1]) # X^(L) @ W^out.T
                    eps_sup         = y_hat - y_batch
                    eps_L           = eps_sup @ weights[-1]
                    errors_extended = errors + [eps_L]


                    # Record batch-averaged energy at this inference step t
                    supervised_energy = 0.5 * eps_sup.pow(2).sum().item() / B
                    latent_energy     = 0.5 * sum(e.pow(2).sum().item() for e in errors) / B
                    batch_supervised_energies.append(supervised_energy)
                    batch_energies.append(latent_energy + supervised_energy)



            # LEARNING - T_learn steps
            with torch.no_grad():           # no autocast - keep precision in weight updates
                for t in range(T_infer + 1, T_learn + T_infer + 1):

                    # Predictions for this learning step t have already been computed
                    # Gradient step for W^(0),...,W^(l-1)
                    for l in range(model.L): # l=0,...,L-1
                        # kernel_modulated_input_latent = kernels[l].to(inputs_latents[l+1].dtype) @ inputs_latents[l+1]
                        # grad_Wl = -(gain_modulated_errors[l].T @ kernel_modulated_input_latent) / B
                        grad_Wl = -(gain_modulated_errors[l].T @ inputs_latents[l+1]) / B
                        weights[l] -= eta_learn * grad_Wl
                    # Gradient step for W^out
                    grad_Wout = eps_sup.T @ inputs_latents[-1] / B
                    weights[-1] -= eta_learn * grad_Wout


                    # Prediction block - for energies at the end of this step t, and for learning at step t+1
                    errors, gain_modulated_errors = model.compute_errors(inputs_latents)
                    y_hat           = model.readout(inputs_latents[-1]) # X^(L) @ W^out.T
                    eps_sup         = y_hat - y_batch


                    # Record batch-averaged energy at this learning step t
                    supervised_energy = 0.5 * eps_sup.pow(2).sum().item() / B
                    latent_energy     = 0.5 * sum(e.pow(2).sum().item() for e in errors) / B
                    batch_supervised_energies.append(supervised_energy)
                    batch_energies.append(latent_energy + supervised_energy)



            # Save this batch’s trajectory to this epoch's list
            epoch_energies.append(batch_energies)
            epoch_supervised_energies.append(batch_supervised_energies)
            
            progress_bar.set_description(f"Batch Energy: {mean(batch_energies):.4f}, Supervised: {mean(batch_supervised_energies):.4f}")
            
        # Save this epoch's list of all batch trajectories
        energy_history.append(epoch_energies)
        supervised_energy_history.append(epoch_supervised_energies)

    return energy_history, supervised_energy_history

@torch.no_grad()
def test_pcn(model, testloader, T_infer, eta_infer, device='cuda'):
    """
    Test a PCN: for each batch, run T_infer inference steps
    updating X^(1),...,X^(L)) exactly as in training, then
    compute Top-1 and Top-3 accuracy on the final readout.
    """
    model.to(device).eval()
    total, top1_correct, top3_correct = 0, 0, 0

    for x_batch, y_batch in tqdm(testloader):
        B = x_batch.size(0)
        total += B
        d_0 = model.dims[0]
        # Flatten inputs
        x_batch = x_batch.view(B, d_0).to(device)
        # Preserve integer labels for metrics
        y_labels = y_batch.to(device)
        # Convert outputs to one-hot vectors for inference
        y_batch = F.one_hot(y_labels, num_classes=model.readout.out_features) \
                       .float().to(device)

        # Concatenate inputs X^(0) and initialized latents X^(l)
        inputs_latents = [x_batch] + model.init_latents(B, device)
        # Weights W^(0),...,W^(L-1),W^out
        weights = [layer.W for layer in model.layers] + [model.readout.weight]

        # INFERENCE - T_infer steps
        with autocast(device_type='cuda'):
            for t in range(1, T_infer + 1):
                errors, gain_modulated_errors = model.compute_errors(inputs_latents)
                y_hat           = model.readout(inputs_latents[-1])
                eps_sup         = y_hat - y_batch
                eps_L           = eps_sup @ weights[-1]
                errors_extended = errors + [eps_L]

                for l in range(1, model.L + 1):
                    grad_Xl = errors_extended[l] - gain_modulated_errors[l-1] @ weights[l-1]
                    inputs_latents[l] -= eta_infer * grad_Xl

        # METRICS
        logits = model.readout(inputs_latents[-1])       # (B, d_out)
        # Top-1
        preds1 = logits.argmax(dim=1)
        top1_correct += (preds1 == y_labels).sum().item()
        # Top-3
        _, preds3 = logits.topk(3, dim=1)                # (B, 3)
        top3_correct += (preds3 == y_labels.unsqueeze(1)).any(dim=1).sum().item()

    acc1 = top1_correct / total
    acc3 = top3_correct / total
    return acc1, acc3


BATCH_SIZE = 500

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2470, 0.2435, 0.2616))
])

trainset = torchvision.datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

trainloader = DataLoader(
    trainset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=True,
    # prefetch_factor=2
)

testset = torchvision.datasets.CIFAR10(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

testloader = DataLoader(
    testset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=True,
    # prefetch_factor=2
)

print(f'Batch size: {BATCH_SIZE}, Train batches: {len(trainloader)}, Test batches: {len(testloader)}')


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

layer_dims = [3072, 1000, 500, 10]
output_dim = 10
pcn_model = PredictiveCodingNetwork(dims=layer_dims, output_dim=output_dim, device=device)
print(pcn_model)

num_epochs = 1
eta_infer = 0.05
eta_learn = 0.005
T_infer = 50
T_learn = BATCH_SIZE # = 500 in this experiment

print("Starting PCN training...")
energy_history, supervised_energy_history = train_pcn(
    model=pcn_model,
    data_loader=trainloader,
    num_epochs=num_epochs,
    eta_infer=eta_infer,
    eta_learn=eta_learn,
    T_infer=T_infer,
    T_learn=T_learn,
    device=device
)

print("\nTraining finished.")


acc1, acc3 = test_pcn(
    model=pcn_model,
    testloader=testloader,
    T_infer=T_infer,
    eta_infer=eta_infer,
    device=device
)
print("")
print(f"Test Top-1 Accuracy: {acc1*100:.2f}%")
print(f"Test Top-3 Accuracy: {acc3*100:.2f}%")


# plot_energy_history_interactive(energy_history, T_infer, T_learn)

# plot_energy_history_interactive(
#     supervised_energy_history,
#     T_infer, T_learn,
#     title='Batch-Averaged Supervised Energy Trajectories'
# )

# plot_epoch_avg_interactive(energy_history, T_infer, T_learn)

# plot_epoch_avg_interactive(
#     supervised_energy_history,
#     T_infer, T_learn,
#     title='Batch-Averaged Supervised Energy Trajectories (Mean ± 1-std)'
# )