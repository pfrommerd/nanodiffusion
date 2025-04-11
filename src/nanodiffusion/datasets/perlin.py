import torch
import math

from nanoconfig import config
from nanoconfig.experiment import Figure, NestedResult
from . import DataConfig, Sample, SampleDataset

def rand_perlin_2d(generator, shape, res, fade = lambda t: 6*t**5 - 15*t**4 + 10*t**3):
    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])

    grid = torch.stack(torch.meshgrid(
        torch.arange(0, res[0], delta[0]),
        torch.arange(0, res[1], delta[1]),
        indexing="ij"
    ), dim = -1) % 1
    angles = 2*math.pi*torch.rand(res[0]+1, res[1]+1, generator=generator)
    gradients = torch.stack((torch.cos(angles), torch.sin(angles)), dim = -1)

    tile_grads = lambda slice1, slice2: gradients[slice1[0]:slice1[1], slice2[0]:slice2[1]].repeat_interleave(d[0], 0).repeat_interleave(d[1], 1)
    dot = lambda grad, shift: (torch.stack((grid[:shape[0],:shape[1],0] + shift[0], grid[:shape[0],:shape[1], 1] + shift[1]  ), dim = -1) * grad[:shape[0], :shape[1]]).sum(dim = -1)

    n00 = dot(tile_grads([0, -1], [0, -1]), [0,  0])
    n10 = dot(tile_grads([1, None], [0, -1]), [-1, 0])
    n01 = dot(tile_grads([0, -1],[1, None]), [0, -1])
    n11 = dot(tile_grads([1, None], [1, None]), [-1,-1])
    t = fade(grid[:shape[0], :shape[1]])
    return math.sqrt(2) * torch.lerp(torch.lerp(n00, n10, t[..., 0]), torch.lerp(n01, n11, t[..., 0]), t[..., 1])

def rand_perlin_2d_octaves(generator, shape, res, octaves=1, persistence=0.5):
    noise = torch.zeros(shape)
    frequency = 1
    amplitude = 1
    for _ in range(octaves):
        noise += amplitude * rand_perlin_2d(generator, shape, (frequency*res[0], frequency*res[1]))
        frequency *= 2
        amplitude *= persistence
    return noise

def generate_perlin_sample(generator, N=64):
    p = rand_perlin_2d_octaves(generator, (128,128), (8,8), 5)
    values = rand_perlin_2d_octaves(generator, (128,128), (8,8), 5)

    p = (p - p.min())
    p = p / p.sum()
    xs, ys = torch.meshgrid(torch.arange(0, 128), torch.arange(0, 128), indexing="xy")
    xs, ys, p, values = xs.flatten(), ys.flatten(), p.flatten(), values.flatten()

    idxs = torch.multinomial(p, N, replacement=False, generator=generator)
    xs, ys, values = xs[idxs], ys[idxs], values[idxs]
    conds = torch.stack((xs, ys), dim=-1).float()
    samples = torch.unsqueeze(values.float(), dim=-1)
    return conds, samples

class PerlinDataset(SampleDataset):
    def __init__(self, conds, samples):
        self.N = conds.shape[0]
        self.conds, self.samples = conds, samples

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        return Sample(self.conds[idx], self.samples[idx])

    def visualize_batch(self, batch : Sample) -> NestedResult:
        import plotly.graph_objects as go
        x = batch.cond[:, 0].cpu().numpy() # type: ignore
        y = batch.cond[:, 1].cpu().numpy() # type: ignore
        z = batch.sample[:, 0].cpu().numpy()
        return Figure(go.Figure(data=go.Scatter(x=x, y=y,
            mode='markers', marker=dict(size=5, color=z,
                    colorscale='Viridis', showscale=True))))

@config(variant="perlin")
class PerlinDataConfig(DataConfig):
    def create(self):
        generator = torch.Generator().manual_seed(0)
        conds, samples = generate_perlin_sample(generator, 4096)
        conds_train, samples_train = conds[:-128], samples[:-128]
        conds_test, samples_test = conds[-128:], samples[-128:]
        train_data = PerlinDataset(conds_train, samples_train)
        test_data = PerlinDataset(conds_test, samples_test)
        return train_data, test_data
