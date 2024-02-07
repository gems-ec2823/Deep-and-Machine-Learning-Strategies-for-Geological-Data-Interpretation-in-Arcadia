
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import os 
import pandas as pd
from IPython.display import display
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import torch.optim as optim
import torch
import torch.nn as nn
import torch.optim as optim
#from livelossplot import PlotLosses
#from sklearn.metrics import confusion_matrix
import numpy as np
import seaborn as sns
import os

import torch
import torch.optim as optim
from livelossplot import PlotLosses
from sklearn.metrics import confusion_matrix
import numpy as np
import seaborn as sns


path_images = '/content/drive/My Drive/core_npy'
path_csv = '/content/drive/My Drive/faciesclass'


def preprocess(npy_file, csv_file):
  img_dict = {
        "204-19-6.png.npy": [2208, 2221],
        "204-19-3a": [1954.5, 2262.3],
        "204-19-3a(1)": [1954.5, 2047],
        "204-19-3a(2)": [2047, 2091],
        "204-19-3a(3)": [2184, 2262.3],
        "204-19-7": [2105, 2551.68],
        "204-19-7(1)": [2105, 2114],
        "204-19-7(2)": [2131, 2180],
        "204-19-7(3)": [2540, 2551.68],
        "204-20-1.png.npy": [1945, 2116],
        "204-20-1Z.png.npy": [2673, 2685],
        "204-20-2.png.npy": [1998, 2018],
        "204-20-3": [2401, 2978.15],
        "204-20-3(1)": [2401, 2436],
        "204-20-3(2)": [2658, 2685.7],
        "204-20-3(3)": [2958, 2978.15],
        "204-20-6a.png.npy": [2175, 2201],
        "204-20a-7.png.npy": [2141, 2161.7],
        "204-24a-6": [2211, 2493.5],
        "204-24a-6(1)": [2211, 2230.6],
        "204-24a-6(2)": [2416, 2493.5],
        "204-24a-7.png.npy": [2075, 2240.65]
    }
  def npy_to_png():
      """
      Convert a .npy file to a .png file, adjusting the height to ensure
      that 0.01 meters corresponds to an integer number of pixels.

      Parameters:
      - npy_path: Path to the .npy file.
      - output_path: Path to save the .png file.
      - total_depth_meters: Total depth in meters that the .npy image represents.
      - height_pixels: Current height in pixels of the .npy image data.
      """
      # Load the .npy file
      npy_path = os.path.join(path_images, npy_file)

      total_depth = img_dict[npy_file][1] - img_dict[npy_file][0]
      npy_data = np.load(npy_path)
      image = Image.fromarray(npy_data.astype(np.uint8))

      height_pixels = image.size[1]

      # Calculate the depth represented by each pixel
      depth_per_pixel = total_depth / height_pixels

      # Calculate the number of pixels corresponding to 0.01 meter
      pixels_per_0_01_meter = 0.01 / depth_per_pixel

      # Resize the image to make 0.01 meter correspond to an integer number of pixels
      pixels_per_segment = int(round(pixels_per_0_01_meter))  # Arrondi à l'entier le plus proche
      #new_height = pixels_per_segment*(total_depth/0.01) # La nouvelle hauteur calculée
      #new_height = int(round(new_height))

      #image = image.resize((image.width, new_height), Image.ANTIALIAS)

      return image, pixels_per_segment, depth_per_pixel


  def crop_core_imageSS(image, pixels_per_segment):
      width = image.size[0]
      new_height = image.size[1]
      segments = []
      for i in range(0, new_height, pixels_per_segment):
          # S'assurer que le segment ne dépasse pas la hauteur de l'image
          if i + pixels_per_segment > new_height:
              segment_height = new_height - i
          else:
              segment_height = pixels_per_segment

          # Découper l'image
          segment = image.crop((0, i, width, i + segment_height))
          segments.append(segment)

      return segments


  import pandas as pd


  def assign_facies_to_segments(segments, facies_data, depth_per_pixel, starting_depth):
      
      # Liste pour stocker les segments avec leurs étiquettes de faciès

    labeled_segments = []
      
      # La profondeur actuelle au début de l'image
    current_depth = starting_depth
      
    for segment in segments:
        # Calculer la profondeur de fin pour le segment actuel
        segment_height = segment.size[1]
        end_depth = current_depth + (segment_height * depth_per_pixel)

        # Trouver tous les faciès qui chevauchent l'intervalle de profondeur du segment
        overlapping_facies = facies_data[
            (facies_data['Start Depth'] < end_depth) &
            (facies_data['End Depth'] > current_depth)
        ]

        # Calculer la proportion de chaque faciès qui se trouve dans le segment
        proportions = []
        for _, row in overlapping_facies.iterrows():
            overlap_start = max(current_depth, row['Start Depth'])
            overlap_end = min(end_depth, row['End Depth'])
            overlap = overlap_end - overlap_start
            proportions.append((row['Facies Class Name'], overlap))

        # Choisissez le faciès qui occupe la plus grande partie du segment
        if proportions:
            facies_label = max(proportions, key=lambda x: x[1])[0]
        else:
            facies_label = 'unlabeled'

        labeled_segment = {'segment': segment, 'facies': facies_label}
        labeled_segments.append(labeled_segment)

        # Mettre à jour la profondeur actuelle pour le prochain segment
        current_depth = end_depth

      
    return labeled_segments


  def crop_center_of_segments(labeled_segments, left_crop = 200 , right_crop = 50):
      # Liste pour stocker les segments recadrés avec leurs étiquettes
      cropped_labeled_segments = []

      for item in labeled_segments:
          segment = item['segment']
          width, height = segment.size

          # Calculer les nouvelles coordonnées pour recadrer la partie centrale
          left = left_crop
          top = 0
          right = width - right_crop
          bottom = height

          # Recadrer l'image
          center_cropped_segment = segment.crop((left, top, right, bottom))

          # Conserver l'étiquette de faciès avec le segment recadré
          cropped_labeled_segment = {'segment': center_cropped_segment, 'facies': item['facies']}
          cropped_labeled_segments.append(cropped_labeled_segment)

      return cropped_labeled_segments
  image, pixels_per_segment, depth_per_pixel = npy_to_png()
# Ensuite, découper l'image en segments
  segments = crop_core_imageSS(image, pixels_per_segment)

  #print(segments)

# Assigner des étiquettes de faciès aux segments
  starting_depth = img_dict[npy_file][0]  # Supposons que c'est votre profondeur de départ
  facies_path = os.path.join(path_csv, csv_file)

  facies_data = pd.read_csv(facies_path)

  labeled_segments = assign_facies_to_segments(segments, facies_data, depth_per_pixel, starting_depth)

  #print(labeled_segments)

# Recadrer le centre des segments
  cropped_labeled_segments = crop_center_of_segments(labeled_segments)

# Finalement, retourner les segments recadrés avec leurs étiquettes
  return cropped_labeled_segments









def total_dataframe():
    df1 = preprocess("204-20-1Z.png.npy", "204-20-1Z.csv")
    df2 = preprocess("204-19-6.png.npy", "204-19-6.csv")
    #df3 = preprocess("204-20-1.png.npy", "204-20-1.csv") ---
    df4 = preprocess("204-20-6a.png.npy", "204-20-6a.csv")
    df5 = preprocess("204-20a-7.png.npy", "204-20a-7.csv")
    #df6 = preprocess("204-24a-7.png.npy", "204-24a-7.csv")---
    df7 = preprocess("204-20-2.png.npy", "204-20-2.csv")
    df = (df1 + df2 + df4 + df5 + df7)
    #print(len(df))
    filtered_segments = [segment for segment in df if segment['facies'] != 'unlabeled']
    df = filtered_segments
    #print(len(df))
    transformed_segments = []
    for segment in df:
        if segment['facies'] == 'bs':
            segment['facies'] = 'nc'
        transformed_segments.append(segment)

    df = transformed_segments
    #print(len(df))



    transformed_segments = []
    for segment in df:
        if segment['facies'] == 'ih':
            segment['facies'] = 'is'
        transformed_segments.append(segment)
    df = transformed_segments
    #print(len(df))


    updated_segments = []

    for segment in df:
        if segment['facies'] == 'nc':
  # Convert the image tensor to a numpy array and calculate the mean pixel value
            image_array = np.array(segment['segment'])
            mean_value = image_array.mean()
            if mean_value <= 70 :
                updated_segments.append(segment)
        else:
            updated_segments.append(segment)
    df = updated_segments
    print(len(df))
  
  #facies_labels = {label: idx for idx, label in enumerate(set(d['facies'] for d in df))}
  #inverse_facies_labels = {v: k for k, v in facies_labels.items()}

    return df




def split_data(df):
  facies_labels = {label: idx for idx, label in enumerate(set(d['facies'] for d in df))}

  class CoreImageDataset():
    def __init__(self, segments, label_map, transform=None):
        self.segments = segments
        self.label_map = label_map
        self.transform = transform

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        segment_info = self.segments[idx]
        image = segment_info['segment'].convert('RGB')  # Convertir en RGB
        label = self.label_map[segment_info['facies']]
        if self.transform:
            image = self.transform(image)
        return image, label

# Assurez-vous que la transformation inclut la conversion en tenseur
  transform = transforms.Compose([
    transforms.Resize((17, 370)),  # Redimensionner si nécessaire
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
  
  dataset = CoreImageDataset(df, facies_labels, transform)
  from torch.utils.data import random_split

  # Taille totale des données
  total_size = len(dataset)

  # Pourcentage de distribution
  train_percent = 0.7
  val_percent = 0.15
  test_percent = 0.15  # Ou vous pouvez calculer comme (1 - train_percent - val_percent)

  # Calcul des tailles
  train_size = int(train_percent * total_size)
  val_size = int(val_percent * total_size)
  test_size = total_size - train_size - val_size  # Pour s'assurer que la somme est égale à total_size

  # Création des ensembles d'entraînement, de validation et de test
  train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

  from torch.utils.data import DataLoader

  # Paramètres des DataLoaders
  batch_size = 32  # Taille de lot, vous pouvez ajuster selon les capacités de votre machine
  shuffle = True   # Mélanger les données pendant l'entraînement

  # DataLoader pour l'ensemble d'entraînement
  train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)

  # DataLoader pour l'ensemble de validation
  val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

  # DataLoader pour l'ensemble de test
  test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
  
  return train_loader, val_loader, test_loader


#################################################################################


def process_data(npy_path, csv_file):

  
  def load_data():
    df = pd.read_csv(csv_file)     
    # Calculate the difference
    #Facies Class Name
    total_depth = np.abs(df['Start Depth'].min() - df['End Depth'].max())
    print(total_depth)

    npy_data = np.load(npy_path)

    image = Image.fromarray(npy_data.astype(np.uint8))
    return image, total_depth    

  def resize_images(image, total_depth):
    width, base_height = image.size # La nouvelle hauteur calculée

    total_depth_meters = total_depth
    height_pixels = image.size[1]
    depth_per_pixel = total_depth_meters / height_pixels

    # Calcul du nombre de pixels correspondant à 0.01 mètre
    pixels_per_0_01_meter = 0.01 / depth_per_pixel
    
    pixels_per_segment = int(round(pixels_per_0_01_meter))  # Arrondi à l'entier le plus proche
    new_height = int(round(pixels_per_segment*(total_depth/0.01))) # La nouvelle hauteur calculée

    image = image.resize((image.width, new_height), Image.ANTIALIAS)
    return image, pixels_per_segment, new_height



  def crop_core_imageSS(image, pixels_per_segment):
      width = image.size[0]
      segments = []
      for i in range(0, new_height, pixels_per_segment):
          # S'assurer que le segment ne dépasse pas la hauteur de l'image
          if i + pixels_per_segment > new_height:
              segment_height = new_height - i
          else:
              segment_height = pixels_per_segment

          # Découper l'image
          segment = image.crop((0, i, width, i + segment_height))
          segments.append(segment)
      print(segments)
      return segments


  def crop_center_of_segments(segments, left_crop = 100, right_crop = 50):
      cropped_segments = []
      for segment_image in segments:
          width, height = segment_image.size

          # Assurez-vous que 'left' et 'right' sont calculés correctement
          left = left_crop
          right = width - right_crop

          # Vérification pour s'assurer que 'right' est plus grand que 'left'
          if right <= left:
              # Ajustement possible : Réduisez 'left_crop' et 'right_crop' ou ignorez le recadrage pour ce segment
              print("Erreur de recadrage : 'right' <= 'left'. Segment ignoré ou ajusté.")
              continue  # Ignorer ce segment ou ajuster 'left' et 'right' comme nécessaire

          top = 0
          bottom = height
          center_cropped_segment = segment_image.crop((left, top, right, bottom))
          cropped_segments.append({'segment': center_cropped_segment})
      #print(cropped_segments)
      return cropped_segments
  
  class CoreImageDataset(torch.utils.data.Dataset):
    def __init__(self, segments, label_map=None, transform=None):
        self.segments = segments
        self.label_map = label_map
        self.transform = transform

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        segment_info = self.segments[idx]
        image = segment_info['segment'].convert('RGB')  # Convertir en RGB
        if self.transform:
            image = self.transform(image)

        if self.label_map is not None and 'facies' in segment_info:
            label = self.label_map[segment_info['facies']]
            return image, label
        else:
            return image
  transform = transforms.Compose([
    transforms.Resize((17, 370)),  # Resize to the desired size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

  image, total_depth = load_data()
  image, pixels_per_segment, new_height = resize_images(image, total_depth)
  image.show()
  segments = crop_core_imageSS(image, pixels_per_segment)
  cropped_segments = crop_center_of_segments(segments)
  print(cropped_segments)
  dataset = CoreImageDataset(cropped_segments, transform=transform)

# Créer le DataLoader
  loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

  return loader



################################################################################################
#################################################################################################
##############################################################################################
######################################### MODEL#############################################
####################################################################################################
###############################################################################################
###############################################################################################


device = 'cpu'
if torch.cuda.device_count() > 0 and torch.cuda.is_available():
    print("Cuda installed! Running on GPU!")
    device = 'cuda'
else:
    print("No GPU available!")




def train2(train_loader, val_loader):

  class SimpleCNN(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.5):
        super(SimpleCNN, self).__init__()
        # Couche de convolution 1
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.dropout1 = nn.Dropout(p=dropout_rate)

        # Couche de convolution 2
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=dropout_rate)

        # Couche de convolution 3
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(p=dropout_rate)

        # Couche de convolution 4
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.relu4 = nn.ReLU()
        self.dropout4 = nn.Dropout(p=dropout_rate)

        # Taille de la sortie de la dernière couche de pooling
        # Vous devrez peut-être ajuster cette taille en fonction de la taille de vos images
        self.fc_input_size = 128 * 23 * 1
        
        # Couche entièrement connectée
        self.fc = nn.Linear(self.fc_input_size, num_classes)

    def forward(self, x):
        # Bloc 1
        x = self.pool(self.relu1(self.bn1(self.conv1(x))))
        x = self.dropout1(x)
        
        # Bloc 2
        x = self.pool(self.relu2(self.bn2(self.conv2(x))))
        x = self.dropout2(x)
        
        # Bloc 3
        x = self.pool(self.relu3(self.bn3(self.conv3(x))))
        x = self.dropout3(x)
        
        # Bloc 4
        x = self.pool(self.relu4(self.bn4(self.conv4(x))))
        x = self.dropout4(x)

        # Aplatir les caractéristiques de la dernière couche de pooling
        x = x.view(-1, self.fc_input_size)
        x = self.fc(x)
        return x

  def train(model, optimizer, criterion, train_loader, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    train_loss = running_loss / len(train_loader)
    train_accuracy = correct / total
    return train_loss, train_accuracy

  def validate(model, criterion, val_loader, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(val_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    val_loss = running_loss / len(val_loader)
    val_accuracy = correct / total
    return val_loss, val_accuracy

  def train_model(model, train_loader, val_loader, device):
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    criterion = nn.CrossEntropyLoss()

    liveloss = PlotLosses()
    for epoch in range(100):
        logs = {}
        train_loss, train_accuracy = train(model, optimizer, criterion, train_loader, device)
        logs['log loss'] = train_loss
        logs['accuracy'] = train_accuracy

        validation_loss, validation_accuracy = validate(model, criterion, val_loader, device)
        logs['val_log loss'] = validation_loss
        logs['val_accuracy'] = validation_accuracy

        liveloss.update(logs)
        liveloss.send()  # ou liveloss.draw()

    return model
  # Supposons que `num_classes` est le nombre de classes de faciès que vous avez
  num_classes = 5  # Remplacez ceci par le nombre réel de classes

  # Initialisation du modèle
  model = SimpleCNN(num_classes=num_classes)

  # Transfert du modèle sur le GPU si disponible

  model.to(device)
  train_model(model, train_loader, val_loader, device)
  return model


def pred(model, test_loader):
  all_preds = []
  all_labels = []

  model.eval()
  with torch.no_grad():
      for inputs, labels in test_loader:
          inputs, labels = inputs.to(device), labels.to(device)
          outputs = model(inputs)
          _, predicted = torch.max(outputs.data, 1)

          all_preds.extend(predicted.cpu().numpy())
          all_labels.extend(labels.cpu().numpy())

  conf_matrix = confusion_matrix(all_labels, all_preds)
  plt.figure(figsize=(10, 7))
  sns.heatmap(conf_matrix, annot=True, fmt='g')
  plt.xlabel('Predicted')
  plt.ylabel('True')
  plt.show()

  return all_preds

#########################################################################





def prediction_unseen_well(loader, model, inverse_facies_labels):
  model.eval()  # Mettre le modèle en mode évaluation
  predictionsSS = []

  with torch.no_grad():
      for image in loader:
          image = image.to(device)  # Assurez-vous que l'image est sur le bon dispositif
          output = model(image)
          pred = output.argmax(dim=1).item()
          predictionsSS.append(pred)

  predicted_labelsSS = [inverse_facies_labels[pred] for pred in predictionsSS]
  return predicted_labelsSS




def plot_prediction(npy_path, csv_file, predicted_labelsSS):
  from PIL import Image
  df = pd.read_csv(csv_file)     
  # Calculate the difference
  total_depth = np.abs(df['Start Depth'].min() - df['End Depth'].max())
  npy_data = np.load(npy_path)

  image = Image.fromarray(npy_data.astype(np.uint8))

  def resize_images(image, total_depth):
    width, base_height = image.size # La nouvelle hauteur calculée

    total_depth_meters = total_depth
    height_pixels = image.size[1]
    depth_per_pixel = total_depth_meters / height_pixels

    # Calcul du nombre de pixels correspondant à 0.01 mètre
    pixels_per_0_01_meter = 0.01 / depth_per_pixel
    
    pixels_per_segment = int(round(pixels_per_0_01_meter))  # Arrondi à l'entier le plus proche
    new_height = int(round(pixels_per_segment*(total_depth/0.01))) # La nouvelle hauteur calculée

    resized_base_image = image.resize((image.width, new_height), Image.ANTIALIAS)
    return resized_base_image, pixels_per_segment

  
  resized_base_image, pixels_per_segment = resize_images(image, total_depth)
  # Créer une image masque pour les prédictions de faciès
  mask_image = Image.new('RGBA', resized_base_image.size, (0, 0, 0, 0))  # Image transparente pour le masque

  # Dictionnaire pour mapper les prédictions aux couleurs
  facies_colors = {
      's': (255, 0, 0, 128),   # red with transparency
      'sh': (0, 255, 0, 128),  # green with transparency
      'nc': (0, 0, 0, 128),    # black with transparency
      'is': (0, 0, 255, 128),  # blue with transparency
      'os': (255, 255, 0, 128) # yellow with transparency
  }

  # Liste des prédictions (exemple : ['nc', 'sh', 'is', ...])
  predictions = predicted_labelsSS  # Votre liste de prédictions

  # Hauteur des segments colorés en pixels
  segment_height = pixels_per_segment

  # Ajouter les couleurs des prédictions sur l'image de base
  y_offset = 0
  for prediction in predictions:
      color = facies_colors[prediction]
      for _ in range(segment_height):  # Repeat for the number of pixels in the segment height
          mask_image.paste(color, [0, y_offset, mask_image.width, y_offset + 1])
          y_offset += 1

  # Composite the mask image onto the resized base image
  superposed_image = Image.alpha_composite(resized_base_image.convert('RGBA'), mask_image)

  final_image = superposed_image.resize(image.size, Image.ANTIALIAS)

  final_image.show()

  # # Sauvegarder l'image résultante si nécessaire
  #final_image.save('/content/drive/My Drive/core_image/superposed_core_well_final.png')

  return final_image











  








