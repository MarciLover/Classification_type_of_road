from torch.utils.data import DataLoader
import settings

def start_dataloader(train_data_2, val_data_2, test_data_2):
    train_dataloader_2 = DataLoader(train_data_2, 
                                    batch_size=settings.BATCH_SIZE,
                                    # num_workers=1,
                                    shuffle=True)

    val_dataloader_2 = DataLoader(val_data_2, 
                                    batch_size=settings.BATCH_SIZE, 
                                    # num_workers=1, 
                                    shuffle=False)

    test_dataloader_2 = DataLoader(test_data_2, 
                                    batch_size=settings.BATCH_SIZE, 
                                    # num_workers=1, 
                                    shuffle=False)
    
    return train_dataloader_2, val_dataloader_2, test_dataloader_2