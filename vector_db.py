import os
import json
import numpy as np
import chromadb

from chromadb import Client, Settings

from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader
# from directional_clip import DirectionalClipTransformEmbeddingFunction



from PIL import Image


class VectorDB:

    def __init__(self, _category='chair'):

        self.label_dict, self.train_uids, self.test_uids, self.train_captions, self.test_captions, self.train_uris, self.test_uris = self.load_human_captions(_category=_category)

        self.clip_ef = OpenCLIPEmbeddingFunction()
        print(f'clip_ef.preprocess: \n{self.clip_ef._preprocess}')
        self.img_loader = ImageLoader()

        self.category_id = '03001627' if _category == 'chair' else '04379243'
        self.img_root = f'/data/ting/ting/text_guided_3D_gen/TAPS3D/ShapeNetCoreRendering/img/{self.category_id}/'

        """
        self.directional_clip_ef = DirectionalClipTransformEmbeddingFunction()
        """

        db_path = "./clip_chroma"
        if os.path.exists(db_path):

            print('Opening persistent client...')

            self.client = chromadb.PersistentClient(path=db_path)

            self.clip_collection = self.client.get_collection(
                name='open-clip',
                embedding_function=self.clip_ef,
                data_loader=self.img_loader
            )
            """
            self.dir_clip_collection = self.client.get_collection(
                name='dir-clip',
                embedding_function=self.directional_clip_ef,
                data_loader=ImageLoader()
            )
            """

        else:

            print('Creating persistent client...')

            self.client = Client(Settings(is_persistent=True, persist_directory=db_path))

            print('Client set up completed...')
            self.clip_collection = self.client.create_collection(
                name='open-clip',
                embedding_function=self.clip_ef,
                data_loader=self.img_loader
            )

            print('Creation of collection completed...')

            """
            self.dir_clip_collection = self.client.create_collection(
                name='dir-clip',
                embedding_function=self.directional_clip_ef,
                data_loader=ImageLoader()
            )
            """

            # API Doc
            # https://colab.research.google.com/github/chroma-core/chroma/blob/main/examples/multimodal/multimodal_retrieval.ipynb#scrollTo=hfFjHW_p2yGG

            # load a subsection of the human labeled data into the database
            metadata = []
            for caption, uid_w_index in zip(self.train_captions, self.train_uid):
                uid = uid_w_index.split('_')[0]
                metadata.append({"uid": uid, "caption": caption})

            print('Preparation of metadata completed...')
            self.clip_collection.add(
                ids=self.train_uids,  # model uids
                metadatas=metadata,  # description of models
                uris=self.train_uris
            )

            print('Data loaded into collection...')

            # print(self.client.get_collection('open-clip').get())


            """
            self.dir_clip_collection.add(
                ids=self.train_uids,  # model uids
                metadatas=self.train_captions,  # description of models
                uris=self.train_uris
            )
            """

    def load_human_captions(self, _category='chair'):

        assert _category == 'chair' or _category == 'table', 'Unsupported category: please select either "chair" or "table". '
        category_id = None
        if _category == 'chair':
            category_id = '03001627'
        elif _category == 'table':
            category_id = '04379243'

        root = '/data/ting/ting/text_guided_3D_gen/clip-captioning-experiment'
        assert os.path.exists(root), f'>> Root directory does not exist. Errorous root path: {root}'
        json_fp = os.path.join(root, 'captions', 'human_captions_shapenet.json')
        assert os.path.isfile(json_fp), f'>> Human caption json file does not exist. Errorous file path: {json_fp}'

        # Opening JSON file
        f = open(json_fp)

        # returns JSON object as
        # a dictionary
        data = json.load(f)

        # get a subset of the json file out for "train"/test split
        training_perc = 0.2
        mask = np.random.rand(len(data["captions"])) <= training_perc
        print(len(mask))
        print(mask[:20])
        print(f'type(data["captions"]): {type(data["captions"])}')
        print(f'len(data["captions"]): {len(data["captions"])}')

        train_uids = []
        test_uids = []

        train_captions = []
        test_captions = []

        train_uris = []
        test_uris = []

        label_dict = {}

        for i in range(len(data["captions"])):
            uid = data["captions"][i]["model"]
            caption_raw = data["captions"][i]["caption"]
            caption = " ".join(caption_raw).replace(" .", ".")

            # check if uid exists
            taps3d_root = '/data/ting/ting/text_guided_3D_gen/TAPS3D'
            model_path = os.path.join(taps3d_root, 'ShapeNetCoreRendering', 'img', category_id, uid, 'models')
            if not os.path.exists(model_path):
                continue
            if uid in label_dict.keys():
                continue

            mode = 'train' if mask[i] else 'test'
            for j in range(24):

                index = str(j).zfill(3)

                # we need to create an unique identifier for each image
                # since each 3D model produces 24 images
                # we simply tag the model's uid with the image index to create an unique identifier
                uid_img_index = f'{uid}_{index}'

                img_fn = index + '.png'

                uri = os.path.join(taps3d_root, 'ShapeNetCoreRendering', 'img', category_id, uid, "models", img_fn)

                if mode == 'train':
                    train_uids.append(uid_img_index)
                    train_captions.append(caption)
                    train_uris.append(uri)
                else:
                    test_uids.append(uid_img_index)
                    test_captions.append(caption)
                    test_uris.append(uri)

            label_dict[uid] = caption

        print(f'len(train_uids): {len(train_uids)}')
        print(f'len(train_captions): {len(train_captions)}')
        print(f'len(test_uids): {len(test_uids)}')
        print(f'len(test_uris): {len(test_uris)}')

        return label_dict, train_uids, test_uids, train_captions, test_captions, train_uris, test_uris

    def retrieve_embedding_from_query(self, _text_query):
        return self.clip_ef.get_text_embeddings(text=_text_query)

    def retrieve_image_from_query(self, _text_query, _n_results=5):
        # Get the text embeddings for the input query
        emb = self.clip_ef.get_text_embeddings(text=_text_query)
        emb = [float(i) for i in emb]

        # Query the collection for similar documents
        result = self.clip_collection.query(
            query_embeddings=[emb],
            n_results=_n_results
        )

        # Extract documents and their metadata
        docs = result['documents'][0]
        descs = result["metadatas"][0]
        list_of_docs = []

        # Combine documents and descriptions into a list
        for doc, desc in zip(docs, descs):
            list_of_docs.append((doc, list(desc.values())[0]))

        return list_of_docs

    def retrieve_images_from_image(self, _img_path, _n_results=5):
        # Get the filename of the uploaded image
        # query_image = np.array(Image.open(_img_path))

        # If a data loader is set for the collection, 
        # you can also query with URIs which reference 
        # data stored elsewhere of the supported modalities
        
        retrieved = self.clip_collection.query(
            query_uris=[_img_path],
            n_results=_n_results
        )
        print(f'retrieved: {retrieved}')

        # Extract documents and their metadatay
        # imgs = retrieved['data'][0]
        ids = retrieved['ids'][0]

        return ids

    def retrieve_embeddings_from_image(self, _img_path):
        img = np.array(Image.open(_img_path))
        emb = self.clip_ef(img)
        print(f'Inside retrieve_embeddings_from_image...\ntype(emb): {type(emb)}')
        return emb


    def get_clip_transformed_caption_from_neighbors(self, _img_path):
        # get clip embedding of the image
        img_emb = self.retrieve_embeddings_from_image(_img_path)

        # get n nearest neighbors
        ids = self.retrieve_images_from_image(_img_path)

        # get labels from neighbor ids
        # get label embeddings of neighbors
        labels = []
        label_embs = []
        img_embs = []
        for _id in ids:

            uid = _id.split('_')[0]
            img_fn = _id.split('_')[-1] + '.png'

            label = self.label_dict[uid]
            labels.append(label)
            print(f'label: {label}')
            label_emb = self.clip_ef([label])
            print(f'type(label_emb): {type(label_emb)}')
            print(f'label_emb len: {len(label_emb)}')
            label_embs.append(label_emb)

            img_path = os.path.join(self.img_root, uid, 'models', img_fn)
            img = np.array(Image.open(img_path))
            print(f'img shape: {img.shape}')
            img_emb = self.clip_ef([img])
            print(f'type(img_emb): {type(img_emb)}')
            print(f'img_emb len: {len(img_emb)}')
            img_embs.append(img_emb)
		
	# get direction of transform
        dir_embs = []
        for label_emb, img_emb in zip(label_embs, img_embs):
            dir_emb = np.array(label_emb) - np.array(img_emb)
            dir_embs.append(dir_emb)

        # dir_embs = self.dir_clip_collection.get(ids=ids)

        # get new embeddings by adding direction of transform to the original embeddings
        for i in range(len(dir_embs)):
            if i == 0:
                sum_emb = dir_embs[i]
            else:
                sum_emb += dir_embs[i]

        avg_dir_emb = sum_emb / len(dir_embs)
        transformed_emb = img_emb - avg_dir_emb

        # predict text caption from embedding
        return transformed_emb


