import torch
import pandas as pd
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from models import BiomassNet
from database import ImageBiomassDataset


def main():

    print("Loading test.csv...")
    df_test = pd.read_csv("csiro-biomass/test.csv")

    # Preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Dataset / Loader
    test_dataset = ImageBiomassDataset(df_test, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    print("Loading checkpoint...")
    model = BiomassNet.load_from_checkpoint("csiro-biomass/checkpoints/best.ckpt")

    print("W example INPUT → OUTPUT:")

    with torch.no_grad():
        img, attrs, _ = next(iter(test_loader))
        print(model(img, attrs))

    model.eval()

    predictions = []
    sample_ids = df_test["sample_id"].tolist()

    print("Predicting...")
    idx = 0

    with torch.no_grad():
        for image, attrs, _ in test_loader:   # NOTE: correct unpacking
            pred_vector = model(image, attrs)  # Shape: (1,5)
            pred_scalar = pred_vector.squeeze().tolist()

            # Extract the correct target based on test.csv row
            target_name = df_test.iloc[idx]["target_name"]
            target_index = ImageBiomassDataset.TARGET_COLS.index(target_name)

            predictions.append(pred_scalar[target_index])
            idx += 1

    # Construct submission file
    df_sub = pd.DataFrame({
        "sample_id": sample_ids,
        "target": predictions
    })

    df_sub.to_csv("sample_submission.csv", index=False)
    print("Saved sample_submission.csv")


if __name__ == "__main__":
    main()
