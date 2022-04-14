import csv
import os
import boto3
os.environ['AWS_DEFAULT_REGION'] = 'us-west-2'
with open('credentials.csv','r') as input:
    next(input)
    reader = csv.reader(input)
    for line in reader:
        access_key_id = line[2]
        secret_access_key = line[3]

image= 'dog.jpg'
client = boto3.client('rekognition', aws_access_key_id= access_key_id, aws_secret_access_key = secret_access_key)
def bb_aws_main(image, N, model_name):
	with open(image, 'rb') as source_image:
		source_bytes = source_image.read()

	response = client.detect_labels(Image={'Bytes': source_bytes}, MaxLabels=3, MinConfidence= 95)
	ann={}
	out= {}
	print(response['Labels'][0]['Instances'][0]['BoundingBox']['Left'])
	ann_data={}
	print(len(response['Labels']))
	for i in range(len(response['Labels'])):
		label_data={}
		point_2D=[]
		label_data['classification_label'] = response['Labels'][i]['Name']
		point_2D.append(response['Labels'][i]['Instances'][0]['BoundingBox']['Left'])
		point_2D.append(response['Labels'][i]['Instances'][0]['BoundingBox']['Width'])
		point_2D.append(response['Labels'][i]['Instances'][0]['BoundingBox']['Height'])
		point_2D.append(response['Labels'][i]['Instances'][0]['BoundingBox']['Top'])
		label_data['point_2D'] = point_2D
		ann_data.setdefault('bounding_box', []).append(label_data)

	ann['annotation_type'] = "labeling"       
	ann['data_filename'] = image
	ann['data_type'] = 'image'
	ann['data_annotation'] = ann_data
	out['annotation'] = ann
	return out
