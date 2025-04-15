import encode_voxel_data
import encode_image_data
import sentense_processing
import vocab
import split_data
import os
import const
import torch_utils

def numericalize_datum(datum, image_vocab, voxel_vocab):
    image_ids = image_vocab.lookup_indices(datum["image_tokens"])
    voxel_ids = voxel_vocab.lookup_indices(datum["voxel_tokens"])
    return {"image_ids": image_ids, "voxel_ids": voxel_ids}

def numericalize_data(data, image_vocab, voxel_vocab):
    for i in range(len(data)):
        numericalized_datum = numericalize_datum(data[i], image_vocab, voxel_vocab)
        data[i]["image_ids"] = numericalized_datum["image_ids"]
        data[i]["voxel_ids"] = numericalized_datum["voxel_ids"]
    return data

def preprocess_data(percent=0.1):
    objects = [obj for obj in [f for f in os.listdir(const.IMAGE_DIRECTORY)] if obj.endswith('.stl')]
    objects = [obj for obj in objects if "0050" not in obj]
    objects = [objects[i] for i in range(int(len(objects) * percent))]

    encoded_voxel_data = encode_voxel_data.encode_voxel_objects(objects)
    encoded_image_data = encode_image_data.encode_image_objects(objects)

    dataset = sentense_processing.build_dataset(objects, encoded_image_data, encoded_voxel_data)
    print(len(dataset))
    # Split the dataset into train, validation, and test sets
    train_data, val_data, test_data = split_data.split_data(dataset)
    print(len(train_data), len(val_data), len(test_data))

    image_vocab, voxel_vocab = vocab.build_vocab(train_data)

    train_data = numericalize_data(train_data, image_vocab, voxel_vocab)
    val_data = numericalize_data(val_data, image_vocab, voxel_vocab)
    test_data = numericalize_data(test_data, image_vocab, voxel_vocab)

    print(len(train_data), len(val_data), len(test_data))

    train_data = torch_utils.to_torch_tensor(train_data, columns=["image_ids", "voxel_ids"])
    test_data = torch_utils.to_torch_tensor(test_data, columns=["image_ids", "voxel_ids"])
    val_data = torch_utils.to_torch_tensor(val_data, columns=["image_ids", "voxel_ids"])

    print(len(train_data), len(val_data), len(test_data))

    train_data_loader = torch_utils.get_data_loader(train_data, const.BATCH_SIZE, shuffle=True)
    valid_data_loader = torch_utils.get_data_loader(val_data, const.BATCH_SIZE)
    test_data_loader = torch_utils.get_data_loader(test_data, const.BATCH_SIZE)

    return objects, image_vocab, voxel_vocab, train_data_loader, valid_data_loader, test_data_loader