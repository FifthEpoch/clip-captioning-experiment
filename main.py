from caption_predictor import CaptionPredictor
from vector_db import VectorDB
import random
import torch

category = 'chair'

vectorstore = VectorDB(category)
caption_predictor = CaptionPredictor()

# test
# 1. retrieve new embedding based on how the image's neighbors transformed
# --- make sure this image is in the test set

test_uris = vectorstore.test_uris
img_path = test_uris[random.randint(0, len(test_uris)-1)]
# img_path = '/home/ting/Desktop/tingsthings/text2shape/TAPS3D/ShapeNetCoreRendering/img/03001627/a69c999cd76e29862f8c68dc62edc38/models/000.png'
print(f'img_path: 		{img_path}')

new_emb = vectorstore.get_clip_transformed_caption_from_neighbors(_img_path=img_path)

predicted_caption = caption_predictor.get_caption_prediction_from_embeddings(
    torch.from_numpy(new_emb).float().to("cuda:0"),
    _is_gpu=True,
    _use_beam_search=True,
    _prefix_length=10)

print(f'predicted_caption: {predicted_caption}')
