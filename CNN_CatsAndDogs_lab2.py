#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd 
from keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
import os
print(os.listdir("/Users/nourrasheed/Downloads/dogs_and_cats"))
import PIL
import os
from PIL import Image


# In[2]:


FAST_RUN = False
IMAGE_WIDTH=300
IMAGE_HEIGHT=300
IMAGE_SIZE=(IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS=3


# In[3]:



dogs_file= os.listdir("/Users/nourrasheed/Downloads/dogs_and_cats/dogs")
#resize the images 
path = r'/Users/nourrasheed/Downloads/dogs_and_cats/dogs'

 #dogs_file, e = os.path.splitext(path+file)


       # if os.path.isfile(path+file):
for file_name in dogs_file: 
    if file_name == '.DS_Store':
        continue 
    print("Processing %s" % file_name)
    im = Image.open(os.path.join("/Users/nourrasheed/Downloads/dogs_and_cats/dogs", file_name))
    output = im.resize((300,300), Image.ANTIALIAS)
    output_file_name = os.path.join("/Users/nourrasheed/Downloads/dogs_and_cats/dogs_resized", file_name)
    output.save(output_file_name ,'JPEG', quality=90)
    print("All done")


# In[3]:


dogs_resized=os.listdir("/Users/nourrasheed/Downloads/dogs_and_cats/dogs_resized")


# In[4]:


counter=0
for item in dogs_resized:
    # Incrementing counter variable to get each item in the list
    counter = counter + 1
print(counter) 


# In[6]:


#resize the cats images 
cats_file= os.listdir("/Users/nourrasheed/Downloads/dogs_and_cats/cats")

path = r'/Users/nourrasheed/Downloads/dogs_and_cats/cats'

for file_name in cats_file: 
    if file_name == '.DS_Store':
        continue 
    print("Processing %s" % file_name)
    im = Image.open(os.path.join("/Users/nourrasheed/Downloads/dogs_and_cats/cats", file_name))
    output = im.resize((300,300), Image.ANTIALIAS)
    output_file_name = os.path.join("/Users/nourrasheed/Downloads/dogs_and_cats/cats_resized", file_name)
    output.save(output_file_name ,'JPEG', quality=90)
    print("All done")


# In[5]:


cats_resized=os.listdir("/Users/nourrasheed/Downloads/dogs_and_cats/cats_resized")


# In[6]:


counter=0
for item in cats_resized:
    # Incrementing counter variable to get each item in the list
    counter = counter + 1
print(counter)


# In[7]:


sample = random.choice(cats_resized)
image = load_img("/Users/nourrasheed/Downloads/dogs_and_cats/cats_resized/"+sample)
plt.imshow(image)
print (image.size)


# In[8]:


sample = random.choice(dogs_resized)
image = load_img("/Users/nourrasheed/Downloads/dogs_and_cats/dogs_resized/"+sample)
plt.imshow(image)
print (image.size)


# In[11]:




categories = []
for filename in dogs_resized:
    category = filename.split('.')[0]
    if category == 'dog':
        categories.append(1)
    else:
        categories.append(0)

df = pd.DataFrame({
    'filename': dogs_resized,
    'category': categories
})


# In[12]:


df.head()


# In[13]:


df.tail()


# In[14]:


df.describe


# In[15]:




categories1 = []
for filename in cats_resized:
    category = filename.split('.')[0]
    if category == 'cat':
        categories1.append(0)
    else:
        categories1.append(1)

df1 = pd.DataFrame({
    'filename': cats_resized ,
    'category': categories1
})


# In[16]:


df1.head()


# In[17]:


df1.info()


# In[18]:


df1.isna().sum()


# In[19]:



for file_name in cats_resized: 
    if file_name == '.DS_Store':
        continue 
    
    im = Image.open(os.path.join("/Users/nourrasheed/Downloads/dogs_and_cats/cats_resized", file_name))
    output_file_name = os.path.join("/Users/nourrasheed/Downloads/dogs_and_cats/train", file_name)
    im.save(output_file_name ,'JPEG', quality=90)
  


# In[20]:




for file_name in dogs_resized: 
    if file_name == '.DS_Store':
        continue 
  
    im = Image.open(os.path.join("/Users/nourrasheed/Downloads/dogs_and_cats/dogs_resized", file_name))
    
    output_file_name = os.path.join("/Users/nourrasheed/Downloads/dogs_and_cats/train", file_name)
    im.save(output_file_name ,'JPEG', quality=90)
  


# In[9]:


train_file=os.listdir("/Users/nourrasheed/Downloads/dogs_and_cats/train")


# In[10]:


categories1 = []
for filename in train_file:
    category = filename.split('.')[0]
    if category == 'cat':
        categories1.append(0)
    else:
        categories1.append(1)

data = pd.DataFrame({
    'filename': train_file ,
    'category': categories1
})


# In[ ]:





# In[20]:


#data=pd.merge(df,df1,how='outer')
#data


# In[ ]:



    


# In[11]:


print(data)


# In[12]:


data.head()


# In[13]:


data.tail()


# In[14]:


data.info()


# In[15]:


data.describe


# In[16]:


data['category'].value_counts().plot.bar()


# In[30]:


#from sklearn.utils import shuffle
#df = shuffle(data)
#df


# In[17]:


from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization

model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax')) # 2 because we have cat and dog classes

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

model.summary()


# In[18]:


from keras.callbacks import EarlyStopping, ReduceLROnPlateau


# # Early Stop
# 
# To prevent over fitting we will stop the learning after 10 epochs and val_loss value not decreased
# 
# 
# 

# In[19]:


earlystop = EarlyStopping(patience=18)


# In[20]:


learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', 
                                            patience=2, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)


# In[21]:


callbacks = [earlystop, learning_rate_reduction]


# In[25]:


data["category"] = data["category"].replace({0: 'cat', 1: 'dog'}) 


# In[26]:


train_df, validate_df = train_test_split(data, test_size=0.20, random_state=42)
train_df = train_df.reset_index(drop=True)
validate_df = validate_df.reset_index(drop=True)


# In[27]:


train_df['category'].value_counts().plot.bar()


# In[28]:


validate_df['category'].value_counts().plot.bar()


# In[29]:


total_train = train_df.shape[0]
total_validate = validate_df.shape[0]
batch_size=15


# In[71]:


import os
import cv2
file_id=train_df['filename'].iloc[0]
print (file_id)
sdir=r'"/Users/nourrasheed/Downloads/dogs_and_cats/train'
file_path=os.path.join(sdir, file_id) # should be full path to the image file
print(file_path)
try:
    image=cv2.imread(file_path)
    shape=image.shape
    print(shape)
except:
    print('Invalid image file')


# In[30]:


train_datagen = ImageDataGenerator(
    rotation_range=15,
    rescale=1./255,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1
)

train_generator = train_datagen.flow_from_dataframe(
    train_df, 
     "/Users/nourrasheed/Downloads/dogs_and_cats/train",
    x_col='filename',
    y_col='category',
    target_size=IMAGE_SIZE,
    class_mode='categorical',
    batch_size=batch_size
)


# In[31]:


validation_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow_from_dataframe(
    validate_df, 
    "/Users/nourrasheed/Downloads/dogs_and_cats/train", 
    x_col='filename',
    y_col='category',
    target_size=IMAGE_SIZE,
    class_mode='categorical',
    batch_size=batch_size
)


# In[32]:


example_df = train_df.sample(n=1).reset_index(drop=True)
example_generator = train_datagen.flow_from_dataframe(
    example_df, 
    "/Users/nourrasheed/Downloads/dogs_and_cats/train", 
    x_col='filename',
    y_col='category',
    target_size=IMAGE_SIZE,
    class_mode='categorical'
)


# In[33]:


plt.figure(figsize=(12, 12))
for i in range(0, 15):
    plt.subplot(5, 3, i+1)
    for X_batch, Y_batch in example_generator:
        image = X_batch[0]
        plt.imshow(image)
        break
plt.tight_layout()
plt.show()


# In[34]:


epochs=3 if FAST_RUN else 50
history = model.fit_generator(
    train_generator, 
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=total_validate//batch_size,
    steps_per_epoch=total_train//batch_size,
    callbacks=callbacks
)


# In[ ]:





# In[35]:


model.save_weights("model1.h5")


# In[38]:


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
ax1.plot(history.history['loss'], color='b', label="Training loss")
ax1.plot(history.history['val_loss'], color='r', label="validation loss")
ax1.set_xticks(np.arange(1, epochs, 1))
ax1.set_yticks(np.arange(0, 1, 0.1))

ax2.plot(history.history['accuracy'], color='b', label="Training accuracy")
ax2.plot(history.history['val_accuracy'], color='r',label="Validation accuracy")
ax2.set_xticks(np.arange(1, epochs, 1))

legend = plt.legend(loc='best', shadow=True)
plt.tight_layout()
plt.show()


# In[42]:


test_filenames = os.listdir("/Users/nourrasheed/Downloads/dogs_and_cats/test")
test_df = pd.DataFrame({
    'filename': test_filenames
})
nb_samples = test_df.shape[0]


# In[43]:


test_gen = ImageDataGenerator(rescale=1./255)
test_generator = test_gen.flow_from_dataframe(
    test_df, 
    "/Users/nourrasheed/Downloads/dogs_and_cats/test", 
    x_col='filename',
    y_col=None,
    class_mode=None,
    target_size=IMAGE_SIZE,
    batch_size=batch_size,
    shuffle=False
)


# In[ ]:





# In[ ]:




