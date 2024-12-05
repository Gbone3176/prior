import torch
from torch.utils.data import DataLoader, Dataset
from prior.encoders.image.resnet import ResNet
from prior.encoders.language.bert import ClinicalBERT
from prior.models.prior import Prior
import pytorch_lightning as pl


# Mock Dataset
class MockDataset(Dataset):
    def __init__(self, num_samples, img_dim=(4, 4), text_dim=(5, 32)):
        self.num_samples = num_samples
        self.img_dim = img_dim
        self.text_dim = text_dim

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Randomly generated images and texts
        image = torch.randn(self.img_dim)  # Example: (3, 224, 224)
        text = torch.randn(self.text_dim)  # Example: (10 sentences, 512 dim each)
        return {"image": image, "text": text}


# Test Function
def test_model():
    # Parameters
    batch_size = 4
    num_samples = 100
    img_dim = (1, 4, 4)
    text_dim = 32

    # Initialize Dataset and DataLoader
    dataset = MockDataset(num_samples=num_samples, img_dim=img_dim, text_dim=text_dim)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    image_encoder = ResNet(name='resnet50')
    text_encoder = ClinicalBERT(pretrained=False)

    # Initialize Model
    model = Prior(image_encoder=image_encoder, text_encoder=text_encoder, gpus=[0], train_dataset=dataset, validation_dataset=dataset)
    # print(model)

    # training Loop
    trainer = pl.Trainer(gpus=[0], max_epochs=1)
    trainer.fit(model,dataloader)

    # for batch_idx, batch in enumerate(dataloader):
    #     output = model(batch)
    #     print(f"Batch {batch_idx}, Loss: {output['loss'].item()}")

if __name__ == '__main__':
    # Run Test
    test_model()