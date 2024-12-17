def main():

    import torch
    import pandas as pd
    import tensorboard

    import settings
    # import plot_random_images
    import custom_dataset
    import load_dataset
    import augmentation
    import dataloader
    import visualization.plot_res_learning as plot_res_learning
    import eval_model
    import model
    import visualization.compare_val_res as compare_val_res
    import visualization.plot_transformed_images as plot_transformed_images

    print(f'RANDOM_SEED = {settings.RANDOM_SEED}')

    selected_prepared_data = load_dataset.form_dataset(show_image=False)

    plot_transformed_images.plot_transformed_images(settings.DIR_WORK_PATH, 
                                                    selected_prepared_data['image_path'],
                                                    selected_prepared_data['label'],
                                                    transform=augmentation.data_transform_check,
                                                    n_samples=3,
                                                    vis_plot = settings.vis_plot)

    train_data = custom_dataset.CustomTensorDataset(settings.DIR_WORK_PATH, 
                                                    selected_prepared_data.query('marked_split == "train"')['image_path'], 
                                                    selected_prepared_data.query('marked_split == "train"')['label'], 
                                                    transform=augmentation.data_transform_train_val)
    print(f'len_tain_data = {len(train_data)}')

    val_data = custom_dataset.CustomTensorDataset(settings.DIR_WORK_PATH, 
                                                    selected_prepared_data.query('marked_split == "val"')['image_path'], 
                                                    selected_prepared_data.query('marked_split == "val"')['label'],
                                                    transform=augmentation.data_transform_train_val)
    print(f'len_val_data = {len(val_data)}')

    val_data_visualization = custom_dataset.CustomTensorDataset(settings.DIR_WORK_PATH, 
                                                    selected_prepared_data.query('marked_split == "val"')['image_path'], 
                                                    selected_prepared_data.query('marked_split == "val"')['label'],
                                                    transform=augmentation.data_transform_visualization)
    print(f'len_val_data = {len(val_data)}')

    test_data = custom_dataset.CustomTensorDataset(settings.DIR_WORK_PATH, 
                                                    selected_prepared_data.query('marked_split == "test"')['image_path'], 
                                                    selected_prepared_data.query('marked_split == "test"')['label'], 
                                                    transform=augmentation.data_transform_test)
    print(f'len_test_data = {len(test_data)}')

    train_dataloader, val_dataloader, test_dataloader = dataloader.create_dataloaders(train_data, val_data, test_data, num_workers = None)

    class_names = train_data.classes
    class_dict = train_data.class_to_idx

    model_1 = model.create_model(class_names)
    model_1 = model.freeze_layers(model_1, num_freeze_layers=settings.NUM_FREEZE_LAYERS)
    model_1_res, learn_time_1, y_pred_val_tensor = model.train_model(model_1, train_dataloader, val_dataloader, class_names)

    pd.DataFrame(model_1_res).to_csv(settings.path_res / 'model_res.csv')
    plot_res_learning.plot_loss_curves(model_1_res)

    dict_results, y_pred_tensor  = eval_model.eval_model(model=model_1,
                                                            test_dataloader=test_dataloader,
                                                            class_names=class_names,
                                                            device=settings.device)

    print(dict_results)

    plot_res_learning.confusion_matrix(y_pred_tensor, 
                                            torch.Tensor(test_data.targets), 
                                            class_names, 
                                            'test_conf_matr')
    plot_res_learning.confusion_matrix(y_pred_val_tensor, 
                                            torch.Tensor(val_data.targets), 
                                            class_names, 
                                            'val_conf_matr')

    compare_val_res.vis_errors(y_pred_val_tensor, val_data_visualization)

if __name__ == '__main__':
    main()
