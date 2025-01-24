class MalariaDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str): Root directory containing species folders.
            transform (callable, optional): Transform to apply to the images.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.data = []

        # Mapping for stages in binary vector
        stage_mapping = {'R': 0, 'T': 1, 'S': 2, 'G': 3}

        # Loading the images from the species directory "specie/img/"
        for species in os.listdir(self.root_dir):
            species_dir = os.path.join(self.root_dir, species, 'img')
            if not os.path.isdir(species_dir):
                continue

            for filename in os.listdir(species_dir):
                if filename.endswith('.jpg'):
                    # Parsing species and stages
                    filepath = os.path.join(species_dir, filename)
                    stage_tag = filename.split('-')[-1].split('.')[0]
                    # For multi-stage cases
                    stages = stage_tag.split('_')

                    # Stage vector for mapping stages
                    stage_vector = [0, 0, 0, 0]
                    for stage in stages:
                        if stage in stage_mapping:
                            # print('stage:', stage)
                            stage_vector[stage_mapping[stage]] = 1
                    self.data.append({'filepath': filepath, 'species': species, 'stages': stage_vector})

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = Image.open(item['filepath']).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, item['species'], item['stages']
