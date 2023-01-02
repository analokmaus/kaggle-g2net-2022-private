from configs import *
from configs_chris import *
import torch

export_path = Path('input/g2net-detecting-continuous-gravitational-waves/wave_banks_180_8192.pt')
cfg = Ds20()
train = pd.read_csv(cfg.train_path).sample(8192)
# dataset = cfg.dataset(
#     df=train,
#     data_dir=cfg.train_dir,
#     transforms=cfg.transforms['test'],
#     **dict(cfg.dataset_params, **dict(
#         signal_amplifier=1, 
#         match_time=True, 
#         positive_p=1.0, 
#         return_mask=True,
#         shift_range=(0, 1),
#         ))
# )
dataset = cfg.dataset(
    df=train,
    data_dir=cfg.train_dir,
    transforms=cfg.transforms['test'],
    preprocess=A.Compose([
        AdaptiveResize(img_size=180), ClipSignal(0, 15), 
        NormalizeSpectrogram('column_wise_sqrt')]),
    match_time=True,
    return_mask=True,
    positive_p=1.0,
    signal_amplifier=1.0,
    shift_range=(0, 1),
    test_stat=Path('input/signal_stat.pickle'),
    test_dir=Path('input/g2net-detecting-continuous-gravitational-waves/test/')
)
loader = D.DataLoader(
    dataset, batch_size=128, shuffle=False,
    num_workers=40, pin_memory=True)

all_masks = []
for _, masks, _ in loader:
    all_masks.append(masks.squeeze(-1))
all_masks = torch.concat(all_masks, dim=0)
print(all_masks.shape)
torch.save(all_masks, export_path)
