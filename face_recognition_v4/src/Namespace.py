class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


if __name__ == '__main__':
    args = Namespace(det=0, embeddings='outputs/embeddings.pickle', flip=0, ga_model='', gpu=0, image_size='112,112',
                     model='../insightface/models/model-y1-test2/model,0', threshold=1.24)
    print(args)
