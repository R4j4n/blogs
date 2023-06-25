+++
author = "Rajan Ghimire"
title = "Vision Transformer (ViT)"
date = "2023-02-06"
description = "ViT from scratch in pytorch"
tags = [
    "Computer Vison",
    "PyTorch",

]
+++

![](https://www.xtrafondos.com/wallpapers/resized/optimus-prime-peleando-4937.jpg?s=large)

Transformers were widely used in the field of natural language processing when they were first developed.  Many researchers have begun using the Transformer architecture in other domains, like computer vision, as a result of Transformers' success in the field of Natural Language Processing (NLP). One such architecture, called the Vision Transformer, was developed by Google Research and Brain Team to tackle the challenge of image classification.
Naturally, you must have prior knowledge of how Transformers function and the issues it addressed in order to grasp how ViT operates. Before delving into the specifics of the ViT, I'll briefly explain how transformers function.
*If you already understand Transformers, feel free to skip ahead to the next section*

**Vanilla Transformer:** <br>

Previously Recurrent Neural Network (RNN) and LSTM were widely used in Natural Language Processing tasks like next word prediction, machine translation, text generation and more. One of the biggest issues with these RNNs, is that they make use of sequential computation. For example: Suppose, we are translating word “How are you?” to any other language. In order for your code to process the word "you", it has to first go through "are" and then "you". And two other issues with RNNs are: 
- Loss of information: For example, it is harder to keep track of whether the subject is singular or plural as you move further away from the subject.
- Vanishing Gradient: when you back-propagate, the gradients can become really small and as a result, your model will not be learning much

To overcome the problem of RNNs, The Transformer was introduced. Transformers are based on attention and don't require any sequential computation per layer, only a single step is needed. The attention is word-to-word mechanism i.e. the attention mechanism finds how much a word in a sentence is related to all words in the sentence, including the word analyzed with itself. Finally, transformers don't suffer from vanishing gradients problems that are related to the length of the sequences.

**Understanding the Transformer Encoder:**<br>
![](https://quantdare.com/wp-content/uploads/2021/11/transformer_arch.png)

**Step 1:  Input Embedding**

First layer in Transformer is the embedding layer. This sub-layer converts the input tokens tokenized by the tokenizer into the vectors of dimension 512. Neural networks learn through numbers so each word must be mapped to a vector with continuous values to represent that word.

**Step 2: Positional Encoding**

Position and order of words in a sentence is vital because position of words in sentence defines the grammar and actual semantics of sentence. Recurrent Neural Network take the order of the word into account as they take a sentence word by word in a sequential order. So, we must input some positional information to the embeddings form the first layer as each word in a sentence simultaneously flows through the Transformer encoder / decoder. The model doesn’t have any sense of order/sequence of each word. To incorporate the order of the word, the concept of positional encoding is used. The positional encoding is done using the sine and the cosine function. 

**Step 3: Multi-Headed Attention**

Multi Head attention is the key feature of the transformer. It is the layer that applies mechanism of Self-attention. Attention is a means of selectively weighting different elements in input data, so that they will have an adjusted impact on the hidden states of downstream layers. The attention mechanisms allow a decoder, while it is generating an output word, to focus more on relevant words or hidden states within the network, and focus less on irrelevant information. 
To achieve self-attention, the positional input embedding is fed into 3 distinct fully connected layers to form query(Q), key(k) and value(V) vectors. Here for Example of query is search text on YouTube or google, key is the video title or article title searched for associated with the query text.
Now the query and key undergo dot product multiplication (QKT) to get the score matrix where highest scores are obtained for those words which are to be given more attention in search. Now, scores are scaled down by dividing it by square root of dimensions of queries and keys (√dk). This is done to have more stable gradients, as multiplying values can have exploding gradient problem. Now we have scaled scores. Now, SoftMax is applied to scaled scores to get probability between 0 to 1 for each word, the higher probability words will get more attention and lesser values will be ignored.
Now the matrix after SoftMax is multiplied with value(V) vector. The higher SoftMax will keep the value of word which the model thinks if of higher relevance and Lower scores will be termed as irrelevant. Now the final output matrix is applied to linear layer to perform further processing.<br>
![](https://production-media.paperswithcode.com/methods/Screen_Shot_2020-07-08_at_12.17.05_AM_st5S0XV.png)

Computing Multi-Head attention: <br>
To make this a multi-headed attention computation the query, key, and value into are split into N vectors before applying self-attention. The split vectors then go through the self-attention process individually. Each self-attention process is called a head. The dimensionality of each head is ‘d_k’ is ( embedding_dim / h) where h is number of heads. Each head produces an output vector that gets concatenated into a single vector before going through the final linear layer.
![](https://blog.scaleway.com/content/images/2019/08/atteq.jpeg)

**Step 4: The Residual Connections, Layer Normalization, and Feed Forward Network**
The output of multi-head attention is added to original positional input embedding. This is called Residual connection. The idea behind residual connection is learning what’s left of (residual), without learning the new representation. 
The output of residual is applied to Layer Normalization. Here we perform layer normalization in order to prevent the values of output from becoming bigger. We have performed a lot of operations which may cause the values of the layer output to become bigger. So, we use layer normalization to normalize back again. 
The output of layer normalization is applied to a feed forward network. The feed forward network consists of a couple of linear layers with Relu activation in between. Point-wise feed forward is used to further process the attention output and giving it a weighted representation.

**The Vison Transformer:**


![](https://amaarora.github.io/images/ViT.png)

We are finally prepared to tackle vision transformers now that we have thoroughly explored the internal operation of transformers.<br>

Applying Transformers on images is a challeng for the following reasons: 
- Images convey significantly more information than words, phrases, or paragraphs do, primarily in the form of pixels.
- Even with current hardware, it would be incredibly challenging to focus on every single pixel in the image.
- Instead, using localized focus was a well-liked substitute.
- In fact CNNs do something very similar through convolutions and the receptive field essentially grows bigger as we go deeper into the model's layers, but Tranformers were always going to be computationally more expensive
  

The general architecture can be easily explained in the following five easy steps:

1. Split images into patches.
2. Obtain the Patch Embeddings, also known as the linear embeddings (representation) from each patch.
3. Each of the Patch Embeddings should have position embeddings and a [cls] token.
4. Get the output values for each of the [cls] tokens by passing each one through a Transformer Encoder.
5. To obtain final class predictions, run the representations of [cls] tokens through an MLP Head.


### Step 1 and Step 2:  PatchEmbedding
* * * 
Splitting an image into fixed-size patches and then linearly embedding each one of them using a linear projection layer is one method we use to obtain patch embeddings from an input image.


![Picture by paper authors (Alexey Dosovitskiy et al.)](https://miro.medium.com/v2/resize:fit:1400/0*kEANaRaJkCPu685t)

However, by employing the 2D Convolution procedure, it is actually possible to combine the two stages into a single step.
If we set the the number of out_channels to 768, and both kernel_size & stride to 16, once we perform the convolution operation (where the 2-D Convolution has kernel size 3 x 16 x 16), we can get the Patch Embeddings matrix of size 196 x 768 like below: [source](https://amaarora.github.io/2021/01/18/ViT.html#the-vision-transformer)
```
# input image `B, C, H, W`
x = torch.randn(1, 3, 224, 224)
# 2D conv
conv = nn.Conv2d(3, 768, 16, 16)
conv(x).reshape(-1, 196).transpose(0,1).shape

>> torch.Size([196, 768])
```
```python
class PatchEmbedding(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = tuple([img_size,img_size])
        patch_size = tuple([patch_size, patch_size])
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

```

### Step 3: CLS TOKEN & Positional Encoding
* * * 
One of the interesting things about the Vision Transformer is that the architecture uses Class Tokens.These Class Tokens are randomly initialized tokens that are prepended to the beginning of your input sequence. The class token has the role of capturing information about the other tokens.<br>

Since the token is randomly initialized, it doesn't have any meaningful data on it by itself. The deeper and more layered the Transformer is,the more information the Class Token can gather from the other tokens in the sequence.<br>

When the Vision Transformer completes the sequence's final classification, it utilizes an MLP head that only considers information from the Class Token of the last layer and no other information. The Class Token appears to be a placeholder data structure that is used to store information that is gleaned from other tokens in the sequence.<br>
**[cls]** token is a vector of size **1 x 768**

![](https://miro.medium.com/v2/resize:fit:828/0*F_igiisSnY9tUeAK)
The positional information of each word within the input sequence is often attempted to be encoded when using transformers to create language models. Each word has a positional encoding that indicates where it should be in the sentence. The Vision Transformer does the same thing by adding a positional encoding to every patch. The top left patch represents the first token, and the bottom right patch represents the last token.
The position embedding is just a tensor of shape $(batchsize,num of patch + 1, embedding size)$ that is added to the projected patches. 

![](https://amaarora.github.io/images/vit-03.png)

so, adding [CLS] token and Positional Encoding to the ```PatchEmbed``` class: 

```python

class PatchEmbedding(nn.Module):
    """ Image to Patch Embedding + cls token + positonal encoding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = tuple([img_size,img_size])
        patch_size = tuple([patch_size, patch_size])
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

        # [cls] token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # positional encoding 
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)

        # Add CLS token to the patch embeddings
        cls_tokens = self.cls_token.expand(B, -1, -1)  
        x = torch.cat((cls_tokens, x), dim=1)

        # Adding POS emmbedding 
        x += self.pos_embed
        return x
```

### Step 4: Transformer Encoder
**Attention Block**
```python
class Attention(nn.Module):
    def __init__(self, dim = 768, num_heads=8, qkv_bias=False, qk_scale=None, dropout=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

```
**Multi-Layer Perceptron Block**
```python 
class MLP(nn.Sequential):
    def __init__(self, emb_size: int, L: int = 4, drop_p: float = 0.):
        super().__init__(
            nn.Linear(emb_size, L * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(L * emb_size, emb_size),
        )
```
**Encoder Block**

```python 
class TransformerEncoderBlock(nn.Module):
    def __init__(self,emb_size: int = 768, drop_p: float = 0., forward_expansion: int = 4,forward_drop_p: float = 0., attn_drp: float = 0.):

        super().__init__()
        self.attention = Attention(dim = emb_size, num_heads=8, qkv_bias=False, qk_scale=None, dropout=attn_drp, proj_drop=0.)
        self.mlp = MLP(emb_size, L=forward_expansion, drop_p=forward_drop_p)

        self.drp_out = nn.Dropout(drop_p) if drop_p > 0. else nn.Identity()
        self.layer_norm_1 =  nn.LayerNorm(emb_size)
        self.layer_norm_2 =  nn.LayerNorm(emb_size)

    
    def forward(self,x):

        x = x + self.drp_out(self.attention(self.layer_norm_1(x)))
        x = x + self.drp_out(self.mlp(self.layer_norm_2(x)))

        return x

```
**Wrapping all:**
```python 
class TransformerEncoder(nn.Sequential):
    def __init__(self, depth: int = 12, **kwargs):
        super().__init__(*[TransformerEncoderBlock(**kwargs) for _ in range(depth)])

```

### Step 5: The classification Head and VIT:

```python 
class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size: int = 768, n_classes: int = 1000):
        super().__init__(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size), 
            nn.Linear(emb_size, n_classes))


class ViT(nn.Sequential):
    def __init__(self,     
                in_channels: int = 3,
                patch_size: int = 16,
                emb_size: int = 768,
                img_size: int = 224,
                depth: int = 12,
                n_classes: int = 1000,
                **kwargs):
        super().__init__(
            PatchEmbedding(in_channels, patch_size, emb_size, img_size),
            TransformerEncoder(depth, emb_size=emb_size, **kwargs),
            ClassificationHead(emb_size, n_classes)
        )
```

## Using ViT on custom dataset.
Requirements: 
```Python
einops==0.6.1
matplotlib==3.6.3
Pillow==9.3.0
torch==1.13.1
torchvision==0.14.1

```
**Downloading the dataset.**  <br>

For this demo i am going to use [Cassava Leaf Disease Classification](https://www.kaggle.com/competitions/cassava-leaf-disease-classification). In order to download the dataset, Navigate to your Kaggle profile and download the kaggle.json. Place the json file in the projects directory.

![kaggle Demo 1 ](/blogs/img/ViT/1.png)
![kaggle Demo 2 ](/blogs/img/ViT/2.png)


```python
import opendatasets as od
data_set_url = "https://www.kaggle.com/competitions/cassava-leaf-disease-classification/data"
od.download(data_set_url)

```

```python
# Import libraries
import os
import cv2
import time
import json
import copy
import pandas as pd
import albumentations as albu
import matplotlib.pyplot as plt
import albumentations as albu

import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


```


```python
BASE_DIR="cassava-leaf-disease-classification/"
TRAIN_IMAGES_DIR=os.path.join(BASE_DIR,'train_images')

train_df=pd.read_csv(os.path.join(BASE_DIR,'train.csv'))

train_df.head()
```

```python
print("Count of training images {0}".format(len(os.listdir(TRAIN_IMAGES_DIR))))
```

    Count of training images 21397



```python
with open(f'{BASE_DIR}/label_num_to_disease_map.json', 'r') as f:
    name_mapping = json.load(f)
    
name_mapping = {int(k): v for k, v in name_mapping.items()}
train_df["class_id"]=train_df["label"].map(name_mapping)
```


```python
name_mapping
```




    {0: 'Cassava Bacterial Blight (CBB)',
     1: 'Cassava Brown Streak Disease (CBSD)',
     2: 'Cassava Green Mottle (CGM)',
     3: 'Cassava Mosaic Disease (CMD)',
     4: 'Healthy'}



**Visualization Utils**


```python
def visualize_images(image_ids,labels):
    plt.figure(figsize=(16,12))
    
    for ind,(image_id,label) in enumerate(zip(image_ids,labels)):
        plt.subplot(3,3,ind+1)
        
        image=cv2.imread(os.path.join(TRAIN_IMAGES_DIR,image_id))
        image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        
        plt.imshow(image)
        plt.title(f"Class: {label}",fontsize=12)
        
        plt.axis("off")
    plt.show()
    

def plot_augmentation(image_id,transform):
    plt.figure(figsize=(16,4))
    
    img=cv2.imread(os.path.join(TRAIN_IMAGES_DIR,image_id))
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    
    plt.subplot(1,3,1)
    plt.imshow(img)
    plt.axis("off")
    
    plt.subplot(1,3,2)
    x=transform(image=img)["image"]
    plt.imshow(x)
    plt.axis("off")
    
    plt.subplot(1,3,3)
    x=transform(image=img)["image"]
    plt.imshow(x)
    plt.axis("off")
    
    plt.show()
    
    
def visualize(images, transform):
    """
    Plot images and their transformations
    """
    fig = plt.figure(figsize=(32, 16))
    
    for i, im in enumerate(images):
        ax = fig.add_subplot(2, 5, i + 1, xticks=[], yticks=[])
        plt.imshow(im)
        
    for i, im in enumerate(images):
        ax = fig.add_subplot(2, 5, i + 6, xticks=[], yticks=[])
        plt.imshow(transform(image=im)['image'])
```

**Dataloader**


```python
# DataSet class

class CassavaDataset(Dataset):
    def __init__(self,df:pd.DataFrame,imfolder:str,train:bool = True, transforms=None):
        self.df=df
        self.imfolder=imfolder
        self.train=train
        self.transforms=transforms
        
    def __getitem__(self,index):
        im_path=os.path.join(self.imfolder,self.df.iloc[index]['image_id'])
        x=cv2.imread(im_path,cv2.IMREAD_COLOR)
        x=cv2.cvtColor(x,cv2.COLOR_BGR2RGB)
        
        if(self.transforms):
            x=self.transforms(image=x)['image']
        
        if(self.train):
            y=self.df.iloc[index]['label']
            return x,y
        else:
            return x
        
    def __len__(self):
        return len(self.df)
```

**Transformations**


```python
train_augs = albu.Compose([
    albu.RandomResizedCrop(height=384, width=384, p=1.0),
    albu.HorizontalFlip(p=0.5),
    albu.VerticalFlip(p=0.5),
    albu.RandomBrightnessContrast(p=0.5),
    albu.ShiftScaleRotate(p=0.5),
    albu.Normalize(    
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],),
    ToTensorV2(),
])

valid_augs = albu.Compose([
    albu.Resize(height=384, width=384, p=1.0),
    albu.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],),
    ToTensorV2(),
])
```


```python
train, valid = train_test_split(
    train_df, 
    test_size=0.1, 
    random_state=42,
    stratify=train_df.label.values
)


# reset index on both dataframes
train = train.reset_index(drop=True)
valid = valid.reset_index(drop=True)

train_targets = train.label.values

# targets for validation
valid_targets = valid.label.values
```


```python
train_dataset=CassavaDataset(
    df=train,
    imfolder=TRAIN_IMAGES_DIR,
    train=True,
    transforms=train_augs
)

valid_dataset=CassavaDataset(
    df=valid,
    imfolder=TRAIN_IMAGES_DIR,
    train=True,
    transforms=valid_augs
)
```


```python
def plot_image(img_dict):
    image_tensor = img_dict[0]
#     print(type(image_tensor))
    target = img_dict[1]
    print(target)
    plt.figure(figsize=(10, 10))
    image = image_tensor.permute(1, 2, 0) 
    plt.imshow(image)
    
plot_image(train_dataset[23])

```

    Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).


    3



    
![Dataloder](/blogs/img/ViT/training_15_2.png)



```python
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    num_workers=4,
    shuffle=True,
)

valid_loader = DataLoader(
    valid_dataset,
    batch_size=32,
    num_workers=4,
    shuffle=False,
)

```

**Train and Valid pipeline**


```python
from tqdm import tqdm
def train_model(datasets, dataloaders, model, criterion, optimizer, scheduler, num_epochs, device):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs-1))
        print('-' * 10)

        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0.0
            
            train_bar = tqdm(dataloaders[phase], desc=f"Training")
            for _, (inputs, labels) in enumerate(train_bar):
                inputs = inputs.to(device)
                labels=labels.to(device)

                # Zero out the grads
                optimizer.zero_grad()

                # Forward
                # Track history in train mode
                with torch.set_grad_enabled(phase == 'train'):
                    model=model.to(device)
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                # Statistics
                running_loss += loss.item()*inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss/len(datasets[phase])
            epoch_acc = running_corrects.double()/len(datasets[phase])

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time()-since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    model.load_state_dict(best_model_wts)
    return model
```

### Loading our custom model.


```python
# importig our custom vit model
from vit import VisionTransformer
model = VisionTransformer(n_classes=len(name_mapping))
```


```python

datasets={'train':train_dataset,'valid':valid_dataset}
dataloaders={'train':train_loader,'valid':valid_loader}
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cuda:1"
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)
criterion=nn.CrossEntropyLoss()
num_epochs=5
```


```python
from tqdm import tqdm
def train_model(datasets, dataloaders, model, criterion, optimizer, scheduler, num_epochs, device):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs-1))
        print('-' * 10)

        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0.0
            
            train_bar = tqdm(dataloaders[phase], desc=f"Training")
            for _, (inputs, labels) in enumerate(train_bar):
                inputs = inputs.to(device)
                labels=labels.to(device)

                # Zero out the grads
                optimizer.zero_grad()

                # Forward
                # Track history in train mode
                with torch.set_grad_enabled(phase == 'train'):
                    model=model.to(device)
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                # Statistics
                running_loss += loss.item()*inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss/len(datasets[phase])
            epoch_acc = running_corrects.double()/len(datasets[phase])

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time()-since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    model.load_state_dict(best_model_wts)
    return model
```


```python
trained_model=train_model(datasets,dataloaders,model,criterion,optimizer,scheduler,num_epochs,device)
```

    Epoch 0/4
    ----------


    Training: 100%|██████████| 602/602 [06:41<00:00,  1.50it/s]


    train Loss: 1.1578 Acc: 0.6166


    Training: 100%|██████████| 67/67 [00:16<00:00,  4.01it/s]


    valid Loss: 1.0497 Acc: 0.6364
    
    Epoch 1/4
    ----------


    Training: 100%|██████████| 602/602 [06:40<00:00,  1.50it/s]


    train Loss: 0.9603 Acc: 0.6489


    Training: 100%|██████████| 67/67 [00:16<00:00,  4.01it/s]


    valid Loss: 0.9217 Acc: 0.6593
    
    Epoch 2/4
    ----------


    Training: 100%|██████████| 602/602 [06:41<00:00,  1.50it/s]


    train Loss: 0.8384 Acc: 0.6853


    Training: 100%|██████████| 67/67 [00:16<00:00,  3.99it/s]


    valid Loss: 0.8747 Acc: 0.6860
    
    Epoch 3/4
    ----------


    Training: 100%|██████████| 602/602 [06:44<00:00,  1.49it/s]


    train Loss: 0.8154 Acc: 0.6935


    Training: 100%|██████████| 67/67 [00:16<00:00,  3.95it/s]


    valid Loss: 0.8452 Acc: 0.6921
    
    Epoch 4/4
    ----------


    Training: 100%|██████████| 602/602 [06:45<00:00,  1.49it/s]


    train Loss: 0.7929 Acc: 0.7020


    Training: 100%|██████████| 67/67 [00:16<00:00,  3.98it/s]

    valid Loss: 0.8299 Acc: 0.6981
    
    Training complete in 34m 58s
    Best val Acc: 0.698131


    



```python
torch.save(model.state_dict(), 'Custom.pt')

```

References:
- https://amaarora.github.io/posts/2021-01-18-ViT.html
- https://towardsdatascience.com/implementing-visualttransformer-in-pytorch-184f9f16f632
- https://medium.com/artificialis/vit-visiontransformer-a-pytorch-implementation-8d6a1033bdc5
- https://medium.com/mlearning-ai/vision-transformers-from-scratch-pytorch-a-step-by-step-guide-96c3313c2e0c
- https://www.kaggle.com/code/abhinand05/vision-transformer-vit-tutorial-baseline