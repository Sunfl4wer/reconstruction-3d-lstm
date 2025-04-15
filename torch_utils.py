import torch
import torch.nn as nn
import const
import vocab

def to_torch_tensor(data, columns):
    for i in range(len(data)):
        for column in columns:
            data[i][column] = torch.from_numpy(data[i][column])
    return data

def collate_fn(batch):
    batch_image_ids = [example["image_ids"] for example in batch]
    batch_voxel_ids = [example["voxel_ids"] for example in batch]
    batch_image_ids = nn.utils.rnn.pad_sequence(batch_image_ids, padding_value=const.PAD_INDEX)
    batch_voxel_ids = nn.utils.rnn.pad_sequence(batch_voxel_ids, padding_value=const.PAD_INDEX)
    batch = {
        "image_ids": batch_image_ids,
        "voxel_ids": batch_voxel_ids,
    }
    return batch

def get_data_loader(dataset, batch_size, shuffle=False):
    data_loader = torch.utils.data.DataLoader(
        dataset=vocab.CustomDataset(dataset),
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=shuffle,
    )
    return data_loader