import torch
import pandas as pd
import tensorboard

import settings
# import plot_random_images
import custom_dataset
import load_dataset
import augmentation
import dataloader
import plot_res_learning
import eval_model
import model
import compare_val_res

path_res = settings.make_dir()

print(f'RANDOM_SEED = {settings.RANDOM_SEED}')

device = "cuda" if torch.cuda.is_available() else "cpu"
selected_prepared_data = load_dataset.form_dataset()

load_dataset.plot_transformed_images(load_dataset.DIR_WORK_PATH, 
                        selected_prepared_data['image_path'],
                        selected_prepared_data['label'],
                        transform=augmentation.data_transform_0, 
                        n=3,
                        seed=settings.RANDOM_SEED)

train_data_2 = custom_dataset.CustomTensorDataset(load_dataset.DIR_WORK_PATH, 
                                                  selected_prepared_data.query('marked_split == "train"')['image_path'], 
                                                  selected_prepared_data.query('marked_split == "train"')['label'], 
                                                  transform=augmentation.data_transform_1)
print(f'len_tain_data = {train_data_2.__len__()}')

test_data_2 = custom_dataset.CustomTensorDataset(load_dataset.DIR_WORK_PATH, 
                                                  selected_prepared_data.query('marked_split == "test"')['image_path'], 
                                                  selected_prepared_data.query('marked_split == "test"')['label'], 
                                                  transform=augmentation.data_transform_2)
print(f'len_test_data = {test_data_2.__len__()}')

val_data_2 = custom_dataset.CustomTensorDataset(load_dataset.DIR_WORK_PATH, 
                                                  selected_prepared_data.query('marked_split == "val"')['image_path'], 
                                                  selected_prepared_data.query('marked_split == "val"')['label'],
                                                  transform=augmentation.data_transform_2)
print(f'len_val_data = {val_data_2.__len__()}')

train_dataloader, val_dataloader, test_dataloader = dataloader.start_dataloader(train_data_2, val_data_2, test_data_2)

class_names = train_data_2.classes
class_dict = train_data_2.class_to_idx

model_1 = model.define_model(class_names)
model_1 = model.freeze_layers(model_1, k=settings.NUM_FREEZE_LAYERS)
model_1_res, learn_time_1, y_pred_val_tensor = model.start_learn(model_1, train_dataloader, val_dataloader, class_names)

pd.DataFrame(model_1_res).to_csv(path_res / 'model_res.csv')
plot_res_learning.plot_loss_curves(model_1_res)

dict_results, y_pred_tensor  = eval_model.eval_model(model=model_1,
                                                        test_dataloader=test_dataloader,
                                                        class_names=class_names,
                                                        device=device)

print(dict_results)

plot_res_learning.confusion_matrix(y_pred_tensor, 
                                        torch.Tensor(test_data_2.targets), 
                                        class_names, 
                                        'test_conf_matr')
plot_res_learning.confusion_matrix(y_pred_val_tensor, 
                                        torch.Tensor(val_data_2.targets), 
                                        class_names, 
                                        'val_conf_matr')

compare_val_res.compare_val(y_pred_val_tensor, val_data_2)

