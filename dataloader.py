from torch.utils.data import DataLoader
import settings

def create_dataloaders(train_data, val_data, test_data, num_workers = None):
    train_dataloader = DataLoader(train_data, 
                                    batch_size=settings.BATCH_SIZE_TRAIN,
                                    # num_workers=num_workers,
                                    shuffle=True)

    val_dataloader = DataLoader(val_data, 
                                    batch_size=settings.BATCH_SIZE_VAL, 
                                    # num_workers=num_workers, 
                                    shuffle=False)

    test_dataloader = DataLoader(test_data, 
                                    batch_size=settings.BATCH_SIZE_TEST, 
                                    # num_workers=num_workers, 
                                    shuffle=False)
    
    return train_dataloader, val_dataloader, test_dataloader