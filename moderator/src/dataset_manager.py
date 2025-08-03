

class DatasetManager:
    def __init__(self) -> None:
        pass
    
    @staticmethod
    def dataset_make(dataset_dir, image_filenames, dataset_captions):
        output_file = open(dataset_dir+'/metadata.jsonl', 'w+')
        for image, dataset_caption in zip(image_filenames, dataset_captions):
            record_str = '{"file_name": "'+image+'", "text": "'+dataset_caption+'"}'
            print(record_str, file=output_file)
        output_file.close()  