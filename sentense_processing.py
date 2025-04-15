import numpy as np
import const

def tokenizer(data, token_length=2):
    return np.asarray([data[i:i+token_length] for i in range(0, len(data), token_length)])

def build_dataset(objects, encoded_image_data, encoded_voxel_data):
    dataset = np.asarray([])

    for obj in objects:
        encoded_image = encoded_image_data[obj]
        encoded_image_str = ""
        for _, m in encoded_image.items():
            encoded_image_str += m
        image_tokens = np.asarray([])
        image_tokens = np.append(image_tokens, const.SOS_TOKEN)
        image_tokens = np.append(image_tokens, tokenizer(data=encoded_image_str, token_length=1))
        image_tokens = np.append(image_tokens, const.EOS_TOKEN)


        encoded_voxel = encoded_voxel_data[obj]
        voxel_tokens = np.asarray([])
        voxel_tokens = np.append(voxel_tokens, const.SOS_TOKEN)
        voxel_tokens = np.append(voxel_tokens, tokenizer(data=encoded_voxel, token_length=2))
        voxel_tokens = np.append(voxel_tokens, const.EOS_TOKEN)
        
        dataset = np.append(dataset, {
            "image": obj,
            "encoded_image": encoded_image_str,
            "encoded_voxel": encoded_voxel,
            "image_tokens": image_tokens,
            "voxel_tokens": voxel_tokens
        })
        
    return dataset