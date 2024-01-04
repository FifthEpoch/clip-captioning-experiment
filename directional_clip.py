import clip
import torch
from PIL import Image

"""
class DirectionalClipTransformEmbeddingFunction:

    def __init__(self, model_name: str = "ViT-B/32", device: str = "cpu"):
        self.device = device  # Store the specified device for model execution
        self.model, self.preprocess = clip.load(model_name, self.device)

    def __call__(self, docs: dict[str, str]):
        # Define a method that takes a list of image file paths (docs) and their corresponding captions (caps) as input

        # Create an empty list to store the text embeddings from the corresponding captions (caps)
        list_of_embeddings = []
        img_paths = list(docs.keys())
        for p in img_paths:
            # Open and load an image from the provided path, resize
            image = Image.open(p)
            image = image.resize((224, 224))
			
			# TODO: how do we retrieve the ideal caption
            # get ideal caption for image
            ideal_cap = docs[p]

            img_embeddings = self.get_img_embeddings(image)
            target_text_embeddings = self.get_text_embeddings(ideal_cap)

            delta_embeddings = target_text_embeddings - img_embeddings

            list_of_embeddings.append(delta_embeddings)

        return list_of_embeddings

    def get_img_embeddings(self, img: Image.Image):
        # Preprocess the image and move it to the specified device
        image_input = self.preprocess(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            # Compute the image embeddings using the CLIP model and convert them to NumPy arrays
            img_embeddings = self.model.encode_image(image_input).cpu().detach().numpy()
        return img_embeddings[0]

    def get_text_embeddings(self, text: str):
        # Define a method that takes a text string as input
        text_token = clip.tokenize(text)  # Tokenize the input text
        with torch.no_grad():
            # Compute the text embeddings using the CLIP model and convert them to NumPy arrays
            text_embeddings = self.model.encode_text(text_token).cpu().detach().numpy()
        return text_embeddings[0]

"""

# Below is WIP
class DirectionalClipTransformEmbeddingFunction(EmbeddingFunction[Union[Documents, Images]]):
    def __init__(self, model_name: str = "ViT-B-32", checkpoint: str = "laion2b_s34b_b79k", category: str = "chair") -> None:
        try:
            import open_clip
        except ImportError:
            raise ValueError(
                "The open_clip python package is not installed. Please install it with `pip install open->
            )
        try:
            self._torch = importlib.import_module("torch")
        except ImportError:
            raise ValueError(
                "The torch python package is not installed. Please install it with `pip install torch`"
            )

        try:
            self._PILImage = importlib.import_module("PIL.Image")
        except ImportError:
            raise ValueError(
                "The PIL python package is not installed. Please install it with `pip install pillow`"
            )

        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name=model_name, pretrained=checkpoint
        )
        self._model = model
        self._preprocess = preprocess
        self._tokenizer = open_clip.get_tokenizer(model_name=model_name)
        
        self.label_dict = self._get_labels(category)
    
    
    def _get_labels(_category='chair'):
        assert _category == 'chair' or _category == 'table', 'Unsupported category: please select either "chair" or "table". '
        category_id = None
        if _category == 'chair':
            category_id = '03001627'
        elif _category == 'table':
            category_id = '04379243'

        root = '/home/ting/Desktop/tingsthings/text2shape/TAPS3D'
        assert os.path.exists(root), f'>> Root directory does not exist. Errorous root path: {root}'
        json_fp = os.path.join(root, 'data', 'human_captions_shapenet.json')
        assert os.path.isfile(json_fp), f'>> Human caption json file does not exist. Errorous file path: {json_fp}'

        # Opening JSON file
        f = open(json_fp)

        # returns JSON object as
        # a dictionary
        data = json.load(f)
        
        # make a dictionary {uid: label}
        label_dict = {}
        for i in range(len(data["captions"])):
            uid = data["captions"][i]["model"]
            caption = " ".join(data["captions"][i]["caption"]).replace(" .", ".")
            label_dict[uid] = caption
        
        return label_dict
        
    
    def _encode_image(self, image: Image) -> Embedding:
        pil_image = self._PILImage.fromarray(image)
        with self._torch.no_grad():
            image_features = self._model.encode_image(
                self._preprocess(pil_image).unsqueeze(0)
            )
            image_features /= image_features.norm(dim=-1, keepdim=True)
            return cast(Embedding, image_features.squeeze().tolist())

    def _encode_text(self, text: Document) -> Embedding:
        with self._torch.no_grad():
            text_features = self._model.encode_text(self._tokenizer(text))
            text_features /= text_features.norm(dim=-1, keepdim=True)
            return cast(Embedding, text_features.squeeze().tolist())

    def __call__(self, input: Union[Documents, Images]) -> Embeddings:
        embeddings: Embeddings = []
        for item in input:
            if is_image(item):
                embeddings.append(self._encode_image(cast(Image, item)))
            elif is_document(item):
                embeddings.append(self._encode_text(cast(Document, item)))
        return embeddings


