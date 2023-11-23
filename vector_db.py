import numpy as np

from chromadb import Client, Settings
from clip_embeddings import ClipEmbeddingsfunction

from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageDataLoader
from directional_clip import DirectionalClipTransformEmbeddingFunction

from PIL import Image


class VectorDB:

    def __init__(self):
        self.client = Client(Settings(is_persistent=True, persist_directory="./clip_chroma"))
        self.clip_ef = OpenCLIPEmbeddingFunction()
        self.clip_collection = self.client.create_collection(
            name='clip',
            embedding_function=self.clip_ef,
            data_loader=ImageDataLoader()
        )
        self.directional_clip_ef = DirectionalClipTransformEmbeddingFunction()
        self.dir_clip_collection = self.client.create_collection(
            name='clip',
            embedding_function=self.directional_clip_ef,
            data_loader=ImageDataLoader()
        )

        # API Doc
        # https://colab.research.google.com/github/chroma-core/chroma/blob/main/examples/multimodal/multimodal_retrieval.ipynb#scrollTo=hfFjHW_p2yGG

        # load a subsection of the human labeled data into the database
        self.clip_collection.add(ids=ids,  # model uids
                      metadatas=human_captions,  # description of models
                      uris=image_uris
                      )
        self.dir_clip_collection.add(
            ids=ids,  # model uids
            metadatas=human_captions,  # description of models
            uris=image_uris
        )

    def retrieve_image_from_query(self, _text_query, _n_results=5):
        # Get the text embeddings for the input query
        emb = self.clip_ef.get_text_embeddings(text=_text_query)
        emb = [float(i) for i in emb]

        # Query the collection for similar documents
        result = self.clip_collection.query(
            query_embeddings=emb,
            include=["documents", "metadatas"],
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
        query_image = np.array(Image.open(_img_path))

        retrieved = self.clip_collection.query(
            query_images=[query_image],
            include=['data', 'ids', 'metadatas'],
            n_results=_n_results
        )

        # Extract documents and their metadata
        imgs = retrieved['data'][0]
        list_of_imgs = []

        # put image into a list
        for img_data in imgs:
            list_of_imgs.append(img_data)

        return list_of_imgs

    def retrieve_embeddings_from_image(self, _img_path):
        img = np.array(Image.open(_img_path))
        return self.directional_clip_ef.get_img_embeddings(img)

    def get_clip_transformed_caption_from_neighbors(self, _img_path):
        # get clip embedding of the image
        img_emb = self.retrieve_embeddings_from_image(_img_path)
        # get n nearest neighbors
        list_of_docs = self.retrieve_images_from_image(_img_path)
        # get direction of transform

        # get new embeddings by adding direction of transform to the original embeddings

        # predict text caption from embedding

