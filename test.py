from pathlib import Path
import os
import settings
import augmentation

"""path_res = Path('C:/pythonProject1/Work_files/Classification_task_py/Results/')
path_res = path_res/ settings.__name__
os.mkdir(path_res)
""" 

# path_res = settings.make_dir()

#print(path_res)

# path_2 = path_res / 'model_res.csv'
# print(path_2)

import re
transform = augmentation.data_transform_2

print(str(transform.__dict__['transforms'][0]))

for i in range(len(transform.__dict__['transforms'])):
    if 'Normazile' in str(transform.__dict__['transforms'][i]):
        print(True)

