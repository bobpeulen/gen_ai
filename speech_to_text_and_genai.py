import ocifs
import numpy as np
import torch
import pandas as pd
import whisper
from tqdm.notebook import tqdm
from sentence_transformers import SentenceTransformer
import pickle
import ffmpeg
import ocifs
from json import loads, dumps
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
import glob
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import os
import oci


# print("Update conda")
# os.system("conda update --all")

###################################################################
################################################################### Get Env variables
###################################################################

JOB_RUN_OCID = os.environ.get("JOB_RUN_OCID")

sub_bucket = os.environ.get("sub_bucket", "video_1")
bucket = "oci://West_BP@frqap2zhtzbe/al_jazeera/" +sub_bucket +"/"
print("full bucket = " + bucket)

###################################################################
################################################################### Get one video from bucket
###################################################################

print("create input folder")

#create local folder (in job)
path_input_locally = "./input_video_al_jazeera/"
#bucket = "oci://West_BP@frqap2zhtzbe/al_jazeera/"

try:       
    if not os.path.exists(path_input_locally):         
        os.makedirs(path_input_locally)    

except OSError: 
    print ('Error: Creating directory of input recording')

#copy recording from bucket to local folder
fs = ocifs.OCIFileSystem()
print(fs.ls(bucket))
fs.invalidate_cache(bucket)
fs.get(bucket, path_input_locally, recursive=True, refresh=True)


###################################################################
################################################################### Cut full video in smaller video frames
###################################################################

#create local folder (in job) for storing 3 minute videos
path_3min_videos = "./3min_videos/"

try:       
    if not os.path.exists(path_3min_videos):         
        os.makedirs(path_3min_videos)    

except OSError: 
    print ('Error: Creating directory of 3min videos')

# Get full file name of the video
#required_video_file = "/home/datascience/input_video_al_jazeera/EN-TEST 10MINS.mxf"
full_name_video = (glob.glob("./input_video_al_jazeera/*"))
required_video_file= full_name_video[0]  #gets first file name. If more, loop through the list

# #split video in 3 minutes. times.text contains all the 3 minutes intervals. Stores in 5min_video/ directory

print("load times.txt file")
with open("./times.txt") as f:
    times = f.readlines()

times = [x.strip() for x in times] 
print(times)


print("start splitting videos")
for time in times:
    starttime = int(time.split("-")[0])
    endtime = int(time.split("-")[1])
    ffmpeg_extract_subclip(required_video_file, starttime, endtime, targetname=str(path_3min_videos) + str(times.index(time)+1)+".mp4")
    

###################################################################
################################################################### Transcribe each video from speech to text.
###################################################################

#load whisper model
model = whisper.load_model("medium")

#create list of number of files
number_of_files = (len([entry for entry in os.listdir(path_3min_videos) if os.path.isfile(os.path.join(path_3min_videos, entry))]))
list_files_number = list(range(1,number_of_files+1))

output = []




for recording in list_files_number:
            
        print()
        print("------------------------------")
        print("Start transcribing recording " + str(recording))
        audio_recording = os.path.join(path_3min_videos, (str(recording) + ".mp4"))

        #transcribe recording
        result = model.transcribe(audio_recording)

        #append result for each recording to list
        output.append([recording, result['text']])

        print("------------------------------")
        print(str(recording) + " is transcribed")
        print("------------------------------")

#all in dataframe
df_transcriptions = pd.DataFrame(output, columns=['recording', 'text'])

#drop all empty rows from dataframe. If video is shorter than lenght frames
df_transcriptions.dropna(inplace = True)

###################################################################
################################################################### GenAI Function
###################################################################

compartment_id = "ocid1.compartment.oc1..aaaaaaaae3n6r6hrjipbap2hojicrsvkzatrtlwvsyrpyjd7wjnw4za3m75q"
CONFIG_PROFILE = "DEFAULT"
config = oci.config.from_file('./config', CONFIG_PROFILE)


def predict_genai(input_text):

    # Service endpoint
    endpoint = "https://generativeai.aiservice.us-chicago-1.oci.oraclecloud.com"

    generative_ai_client = oci.generative_ai.GenerativeAiClient(config=config, service_endpoint=endpoint, retry_strategy=oci.retry.NoneRetryStrategy(), timeout=(10,240))

    #full_input_prompt = "Get the top 2 key words summarizing the following text. Only output the key words. Use the following text. Text: " + str(input_text)
    prompts = [input_text]
    generate_text_detail = oci.generative_ai.models.GenerateTextDetails()
    generate_text_detail.prompts = prompts
    generate_text_detail.serving_mode = oci.generative_ai.models.OnDemandServingMode(model_id="cohere.command")
    # generate_text_detail.serving_mode = oci.generative_ai.models.DedicatedServingMode(endpoint_id="custom-model-endpoint") # for custom model from Dedicated AI Cluster
    generate_text_detail.compartment_id = compartment_id
    generate_text_detail.max_tokens = 400
    generate_text_detail.temperature = 0.75
    generate_text_detail.frequency_penalty = 1.0
    generate_text_detail.top_p = 0.7

    generate_text_response = generative_ai_client.generate_text(generate_text_detail)

    # Print result
    print("**************************Generate Texts Result**************************")
    #print(generate_text_response.data)
    print("Response from GenAI = " + generate_text_response.data.generated_texts[0][0].text)
    output_text = generate_text_response.data.generated_texts[0][0].text

    return output_text

###################################################################
################################################################### Apply GenAI Function to each transcribed recording
###################################################################

start_time = 0
#empty list. Empty for interation
output_list = []

df = pd.DataFrame(columns=['recording', 'start_time', 'end_time', 'original_text', 'key_phrases', 'summary'])

for index, row in df_transcriptions.iterrows():
    
    #get values
    input_text = row['text']#.astype("string")
    recording = row['recording']
    
    #create full prompt
    full_input_prompt_key = "Get the top 2 key words summarizing the following text. Only output the key words. Use the following text. Text: " + str(input_text)
    full_input_prompt_summary = "Summarize the following text in maximum 2 sentences. Text: " + str(input_text)

    #run each text to GenAI service
    output_text_key_phrases = predict_genai(full_input_prompt_key)
    output_text_key_summary = predict_genai(full_input_prompt_summary)
    
    #create timestamps
    end_time = start_time + 180

    output_list.append([recording, start_time, end_time, input_text, output_text_key_phrases, output_text_key_summary])
          
    #update start_time
    start_time += 180
    
df_out = pd.DataFrame(output_list, columns=['recording', 'start_time', 'end_time', 'original_text', 'key_phrases', 'summary'])

#replace empty value if any
df_out['original_text'] = df_out['original_text'].replace('', 'no_text').fillna(df_out['original_text'])
df_out = df_out.fillna("no_text")

print("GenAI inferences are done")

###################################################################
################################################################### Save output to bucket as csv
###################################################################

sub_bucket = os.environ.get("sub_bucket", "video_1")
bucket = "oci://West_BP@frqap2zhtzbe/al_jazeera/" + sub_bucket +"/"

output_bucket = bucket + "output.csv"
df_out.to_csv(output_bucket, index=False)

###################################################################
################################################################### Push results to qdrant
###################################################################

#filter on columns to be sure. if indexes go wrong
df_out = df_out[['recording','start_time','end_time','original_text','key_phrases','summary']]
df_out = df_out.fillna("no_text")

print("Start qdrant")

#convert pd dataframe to json
import json
texts = df_out.to_json(orient='records')
documents = loads(texts)

print("define encoder")

#define encoder
from sentence_transformers import SentenceTransformer
encoder = SentenceTransformer('multi-qa-distilbert-cos-v1') # Model to create embeddings

#establish connection to Qdrant vector database
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from qdrant_client import AsyncQdrantClient, models

qdrant = QdrantClient("207.211.188.211", port=6333) # 

print("create collection")

# Create collection to store
qdrant.recreate_collection(
    collection_name="collection_v1",
    vectors_config=models.VectorParams(
        size=encoder.get_sentence_embedding_dimension(), # Vector size is defined by used model
        distance=models.Distance.COSINE
    )
)


# Let's vectorize descriptions and upload to qdrant
qdrant.upload_records(
    collection_name="collection_v2",
    records=[
        models.Record(
            id=idx,
            vector=encoder.encode(doc["original_text"]).tolist(),
            payload=doc
        ) for idx, doc in enumerate(documents)
    ]
)

print('Done. Results to csv and to Qdrant')
