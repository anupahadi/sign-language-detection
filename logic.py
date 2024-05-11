
def assign_layer(self):
      model_embed = self.embed_dict[self.architecture]()

      return model_embed

def obtain_children(self):
      model_embed = nn.Sequential(*list(self.model.children())[:-1])

      return model_embed

def obtain_classifier(self):
      self.model.classifier = self.model.classifier[:-1]

def assign_transform(self, weights):
      weights_dict = {
            "resnet50": models.ResNet50_Weights,
            "vgg19": models.VGG19_Weights,
            "efficientnet_b0": models.EfficientNet_B0_Weights,
       }

       # try load preprocess from torchvision else assign default
      try:
             w = weights_dict[self.architecture]
            weights = getattr(w, weights)
            preprocess = weights.transforms()
       except Exception:
            preprocess = transforms.Compose(
                [
                 transforms.Resize(224),
                 transforms.ToTensor(),
                 transforms.Normalize(
                 mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                 ),
                ]
            )

       return preprocess

# 

def embed_image(self, img):
      # load and preprocess image
      img = Image.open(img)
      img_trans = self.transform(img)

      # store computational graph on GPU if available
      if self.device == "cuda:0":
          img_trans = img_trans.cuda()

      img_trans = img_trans.unsqueeze(0)

      return self.embed(img_trans)

def similar_images(self, target_file, n=None):
      """
      Function for comparing target image to embedded image dataset

      Parameters:
      -----------
      target_file: str specifying the path of target image to compare
            with the saved feature embedding dataset
      n: int specifying the top n most similar images to return
      """

      target_vector = self.embed_image(target_file)

      # initiate computation of consine similarity
      cosine = nn.CosineSimilarity(dim=1)

      # iteratively store similarity of stored images to target image
      sim_dict = {}
      for k, v in self.dataset.items():
          sim = cosine(v, target_vector)[0].item()
          sim_dict[k] = sim

      # sort based on decreasing similarity
      items = sim_dict.items()
      sim_dict = {k: v for k, v in sorted(items, key=lambda i: i[1], reverse=True)}

      # cut to defined top n similar images
      if n is not None:
          sim_dict = dict(list(sim_dict.items())[: int(n)])

      self.output_images(sim_dict, target_file)

      return sim_dict


def cluster_dataset(self, nclusters, dist="euclidean", display=False):
       vecs = torch.stack(list(self.dataset.values())).squeeze()
       imgs = list(self.dataset.keys())
       np.random.seed(100)

       cluster_ids_x, cluster_centers = kmeans(
           X=vecs, num_clusters=nclusters, distance=dist, device=self.device
        )

       # assign clusters to images
       self.image_clusters = dict(zip(imgs, cluster_ids_x.tolist()))

       # store cluster centres
       cluster_num = list(range(0, len(cluster_centers)))
       self.cluster_centers = dict(zip(cluster_num, cluster_centers.tolist()))

       if display:
           self.display_clusters()

       return