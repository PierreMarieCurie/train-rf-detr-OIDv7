from rfdetr import RFDETRBase
import os

def fine_tune_RFDETR(dataset_path, results_path):
    
    # Load pretrained model
    model = RFDETRBase()

    # Capture training metrics after each epoch
    history = []
    def callback2(data):
        history.append(data)
    model.callbacks["on_fit_epoch_end"].append(callback2)
    
    # Fine-tuning
    model.train(dataset_dir=dataset_path,
        epochs=10,
        batch_size=8,
        grad_accum_steps=2,
        lr=1e-4,
        output_dir=results_path
    )