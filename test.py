import csv
import os
import torch
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms


def test():
    data_transform = transforms.Compose([transforms.RandomResizedCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = torch.load('VGG11_model.pth').to(device)
    model.eval()

    classes = os.walk('./train').__next__()[1]
    test_imgs = os.walk('./test').__next__()[2]

    with open('submission.csv', 'w', newline='') as file:
        fieldnames = ['file', 'species']

        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        with torch.no_grad():
            for test_img in test_imgs:
                img_path = os.path.join("./test", test_img)

                image = Image.open(img_path).convert('RGB')
                image = data_transform(image).unsqueeze(0)
                x = Variable(image.to(device))
                y = model(x)
                _, pred = torch.max(y.data, 1)

                writer.writerow({'file' : test_img, 'species' : classes[pred[0]]})
                print(test_img, classes[pred[0]])

if __name__ == '__main__':
    test()
