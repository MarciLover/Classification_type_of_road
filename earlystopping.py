class EarlyStopping:
    def __init__(self, tolerance, min_delta):

        self.tolerance = tolerance
        self.min_delta = min_delta
        self.counter = 0
        self.early_stop = False

    def __call__(self, delta, prev_delta):
        # if (validation_loss - train_loss) <= self.min_delta:
        #    self.counter +=1
        
        if delta <= self.min_delta:
            self.counter +=1
            if self.counter >= self.tolerance:  
                self.early_stop = True
        # elif (validation_loss - train_loss) > self.min_delta:
        #   self.counter = 0
        elif delta > prev_delta:
            self.counter = 0