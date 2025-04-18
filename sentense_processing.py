import numpy as np
import const


def tokenizer(data, token_length=2):
    return np.asarray([data[i:i+token_length] for i in range(0, len(data), token_length)])

def build_dataset(objects, encoded_image_data, encoded_voxel_data):
    dataset = np.asarray([])

    data = {}
    image_token_len = 0
    voxel_token_len = 0

    for obj in objects:
        encoded_image = encoded_image_data[obj]
        encoded_image_str = np.asarray([])
        for _, m in encoded_image.items():
            encoded_image_str = np.append(encoded_image_str, m)
        data[obj] = {
            "image_tokens": encoded_image_str,
            "voxel_tokens": encoded_voxel_data[obj]
        }
        image_token_len = max(image_token_len, len(encoded_image_str))
        voxel_token_len = max(voxel_token_len, len(encoded_voxel_data[obj]))
        
    for obj in objects:
        it = data[obj]["image_tokens"]
        it = np.append(it, np.asarray([const.PAD_TOKEN] * (image_token_len - len(it))))
        image_tokens = np.asarray([const.SOS_TOKEN])
        image_tokens = np.append(image_tokens, it)
        image_tokens = np.append(image_tokens, const.EOS_TOKEN)


        encoded_voxel = data[obj]["voxel_tokens"]
        encoded_voxel = np.append(encoded_voxel, np.asarray([const.PAD_TOKEN] * (voxel_token_len - len(it))))
        voxel_tokens = np.asarray([const.SOS_TOKEN])
        voxel_tokens = np.append(voxel_tokens, encoded_voxel)
        voxel_tokens = np.append(voxel_tokens, const.EOS_TOKEN)
        
        dataset = np.append(dataset, {
            "image": obj,
            "encoded_image": encoded_image_str,
            "encoded_voxel": encoded_voxel,
            "image_tokens": image_tokens,
            "voxel_tokens": voxel_tokens
        })
        
    return dataset