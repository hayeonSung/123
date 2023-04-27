from fastapi import FastAPI, Request, File, UploadFile
import engine.ekgEngine as ekgEngine
from utils.s3Util import s3, s3_download_file, s3_upload_file
import datetime
import pytz
import boto3
import os
import json
from glob import glob
import shutil
from pydantic import BaseModel
api = FastAPI()


@api.get('/')
def root():
    return {'message': 'Hello friends!'}


# @api.post('/file_info')
# async def file_info():
#     bucket_name = 'ignites-ekg-files-dev.synergyai.co'
#     bucket = s3.Bucket(bucket_name)
#     os.makedirs('./temp/', exist_ok=True)

#     downloaded_files = []
#     for s3_object in bucket.objects.all():
#         bucket_object_key = s3_object.key
#         print(bucket_object_key)

#         print(s3_object)
#         local_temp_save_path = './temp/' + bucket_object_key

#         os.makedirs('/'.join(local_temp_save_path.split('/')[:4]), exist_ok=True)
#         file_loaded = s3_download_file(bucket_name, bucket_object_key, local_temp_save_path)
#         downloaded_files.append(bucket_object_key)

#     output = {
#         "Filename": {
#             "ID": "hi",
#             "NAME": "hi",
#             "AGE": "40",
#             "SEX": "M",
#         }
#     }

#     return output
class Item(BaseModel):
    bucket_sub_path: str

@api.post('/predict')
def predict(item: Item):
    dic = item.dict()
    # s3 하위 디렉토리 지정
    bucket_sub_path = dic['bucket_sub_path']
    print(f"reqeust body: {dic}")
    bucket_name = 'ignites-ekg-files-dev.synergyai.co'
    bucket = s3.Bucket(bucket_name)
    os.makedirs('./temp/', exist_ok=True)

    downloaded_files_to_delete = []
    downloaded_files_for_inference = []

    bucket_sub_path = "/".join(bucket_sub_path.split("/")[:2])

    downloaded_files_to_delete = []
    downloaded_files_for_inference = []

    for s3_object in bucket.objects.all():
        bucket_object_key = s3_object.key
        print(bucket_object_key)
        local_temp_save_path = './temp/' + bucket_object_key
        print(local_temp_save_path)
        os.makedirs('/'.join(local_temp_save_path.split('/')[:4]), exist_ok=True)
        file_loaded = s3_download_file(bucket_name, bucket_object_key, local_temp_save_path)
        print(file_loaded)
        downloaded_files_to_delete.append(bucket_object_key)
        downloaded_files_for_inference.append(local_temp_save_path)
    



    #for deleting items in s3
    client = boto3.client('s3')
    for files in downloaded_files_to_delete:
        client.delete_object(Bucket=bucket_name, Key=files)

    KST = pytz.timezone('Asia/Seoul')
    whole_time = str(datetime.datetime.now(KST))
    date_time = whole_time.replace(' ', '_')
    folder_name = date_time[:10]
    date_and_time = date_time[:19]
    upload_bucket_name = 'ignites-ekg-all-uploaded-files.synergyai.co'

    for files in downloaded_files_for_inference:

        file_name = '/'.join(files.split('/')[2:4]) + '/'+date_and_time+'_'+ files.split('/')[-1]

        upload_result = s3_upload_file(files, upload_bucket_name, file_name)



    for input_dir in downloaded_files_for_inference:
        print(input_dir)
        input_dir = input_dir
        model_t2 = './_assets/weights/EKG_resnet18_2022_10_14_0.8407f1_epoch8.pth'
        model_t7 = './_assets/weights/EKG_resnet18_2022_10_07_0.8663f1_epoch9.pth'
        model_t14 = './_assets/weights/EKG_resnet18_2022_10_13_0.8667f1_epoch7.pth'
        model_t30 = './_assets/weights/EKG_resnet18_2022_11_25_0.8587f1_epoch7.pth'

        ekg_engine = ekgEngine.EkgEngine(model_t2)
        output_t2, pinfo = ekg_engine.run(input_dir, batch_size=1, num_workers=4, beat_max_length=700)

        ekg_engine = ekgEngine.EkgEngine(model_t7)
        output_t7, pinfo = ekg_engine.run(input_dir, batch_size=1, num_workers=4, beat_max_length=700)

        ekg_engine = ekgEngine.EkgEngine(model_t14)
        output_t14, pinfo = ekg_engine.run(input_dir, batch_size=1, num_workers=4, beat_max_length=700)

        ekg_engine = ekgEngine.EkgEngine(model_t30)
        output_t30, pinfo = ekg_engine.run(input_dir, batch_size=1, num_workers=4, beat_max_length=700)

        print(output_t2,'t2')
        print(output_t7, 't7')
        print(output_t14, 't14')
        print(output_t30, 't30')
        #output = json.dumps(str(output))

        shutil.rmtree('./temp')

        break
    file_key = input_dir
    print(file_key, 'file key')
    file_name = file_key.replace('./temp/', '')

    prob_t2 = str(round(output_t2[file_key]['output']['class1_prob'] * 100, 1))
    prob_t7 = str(round(output_t7[file_key]['output']['class1_prob'] * 100, 1))
    prob_t14 = str(round(output_t14[file_key]['output']['class1_prob'] * 100, 1))
    prob_t30 = str(round(output_t30[file_key]['output']['class1_prob'] * 100, 1))


    prob_t14 = output_t14[file_key]['output']['class1_prob']
    threshold_t14 = output_t14[file_key]['output']['class1_thres']

    if prob_t14 > threshold_t14:
        risk = "HIGH"
    else:
        risk = "LOW"

    output = {
            "ID": pinfo["ID"],
            "NAME": pinfo["NAME"],
            "AGE": pinfo["AGE"],
            "SEX": pinfo["SEX"],
            "DATETIME": pinfo["DATETIME"],
            "RISK": risk,
    }
    print(output)

    return output
