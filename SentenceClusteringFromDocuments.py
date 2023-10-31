#!/usr/bin/env python
# coding: utf-8

# In[6]:


# Import text from Excel worksheet.

import pandas as pd

file_path = "SampleDocuments2.xlsx"
file_df = pd.read_excel(file_path)
file_df


# In[2]:


# Only needs to run if package not already installed.

get_ipython().system('pip install spacy')


# In[3]:


# Only needs to run if not already installed.

get_ipython().system('python -m spacy download en_core_web_sm')


# In[7]:


# Split text in each cell into lists with separate sentences.

import spacy

nlp = spacy.load("en_core_web_sm")
file_df["doc_text"] = file_df["doc_text"].apply(lambda x: [sent.text for sent in nlp(x).sents])
file_df


# In[8]:


# Put each sentence into a separate row in the dataframe.

file_df = file_df.explode("doc_text", ignore_index=True)

# Update the column headings.

file_df.rename(columns={"doc_text": "sent_text"}, inplace=True)
file_df.index.name = "sent_id"

# Show the contents of the dataframe.

file_df


# In[9]:


# Add the sentence text and IDs to lists for further processing.

sent_id = file_df.index.values.tolist()
sent_text = file_df.sent_text.values.tolist()
sent_id


# In[12]:


# Only needs to run if not already installed.

get_ipython().system('pip install sentence-transformers')


# In[10]:


# Select the model to use for sentence embeddings.

from sentence_transformers import SentenceTransformer

# Models - https://huggingface.co/models?library=sentence-transformers
model = SentenceTransformer('all-mpnet-base-v2')


# In[11]:


# Create the sentence embeddings

embeddings1 = model.encode(sent_text)


# In[12]:


# clustering https://www.youtube.com/watch?v=OlhNZg4gOvA time index 22:00

from sklearn.cluster import KMeans
import numpy as np

# normalize the embeddings to unit length
embeddings_norm = embeddings1 / np.linalg.norm(embeddings1, axis=1, keepdims=True)

# Show the embeddings dataframe.
embeddings_norm


# In[13]:


# Import the libraries needed to create the elbow diagram.

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


# In[ ]:


# Look at the elbow diagram to help determine appropriate number of clusters to create.

# determining the maximum number of clusters
# using the simple method
limit = file_df.shape[0]
 
# selecting optimal value of 'k'
# using elbow method
 
# wcss - within cluster sum of
# squared distances
wcss = {}
 
for k in range(2,limit+1):
    model = KMeans(n_clusters=k)
    model.fit(embeddings_norm)
    wcss[k] = model.inertia_
     
# plotting the wcss values
# to find out the elbow value
plt.plot(wcss.keys(), wcss.values(), 'gs-')
plt.xlabel('Values of "k"')
plt.ylabel('WCSS')
plt.show()


# In[ ]:


# Create the desired number of clusters. (Update the value for ClusterCount.)

ClusterCount = 3

clustering_model = KMeans(n_clusters=ClusterCount)
clustering_model.fit(embeddings_norm)
cluster_assignment = clustering_model.labels_
print(cluster_assignment)


# In[ ]:


# Add the cluster assignments to the dataframe in a new column.

file_df['cluster3'] = cluster_assignment
file_df


# In[ ]:


# Get file path in preparation for saving output to Excel.

import os

OutputFile = os.path.split(file_path)[0] + "\\Clusters_" + os.path.split(file_path)[1]
OutputFile


# In[ ]:


# Save the dataframe to an Excel workbook in the same folder as the original file.

with pd.ExcelWriter(OutputFile) as writer:
    
    # Write the scores dataframe to the Excel workbook. Leave blank rows at the top.
    file_df.to_excel(writer, sheet_name="Sheet1", startrow=0, startcol=0)
    
print('Done.')


# In[ ]:




