# Torch
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import optim
# vision
from torchvision import models
from torchvision import transforms
from torchvision.ops import nms
import selectivesearch
# PIL
from PIL import Image
# Numpy
import numpy as np
import skimage
# matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as mpatches
# XML
import xml.etree.ElementTree as ET
# OS
import os

# GLOBALS
trainDir  = "VOCTrainVal/VOCdevkit/VOC2007/"
testDir   = "VOCTest/VOCdevkit/VOC2007/"
trainAnno = trainDir + "Annotations/"
testAnno  = testDir + "Annotations/"

transform = transforms.Compose([
                                transforms.ToTensor(),
])
resize = transforms.Compose([
			     transforms.ToPILImage(),
			     transforms.Resize((64,64)),
			     transforms.ToTensor(),
                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                  std=[0.229, 0.224, 0.225]),
])

def getClasses(location):
    classes = []
    annotations = os.listdir(location)
    for anno in annotations:
        tree = ET.parse(location + anno)
        root = tree.getroot()
        for child in root.findall('object'):
            name = child.find('name').text
            if name not in classes:
                classes.append(name)

    return classes
classes = getClasses(testAnno)

def getLabelsInImg(filename):
    classList = []
    tree = ET.parse(filename)
    root = tree.getroot()
    for child in root.findall('object'):
        name = child.find('name').text
        classList.append(name)

    return classList

def getGTs(location):
    gt = []
    tree = ET.parse(location)
    root = tree.getroot()
    for child in root.findall('object'):
        name = child.find('name').text
        for gc in child:
            if gc.tag == "bndbox":
                for ggc in gc:
                    if ggc.tag == "xmin":
                        xmin = int(ggc.text)
                    if ggc.tag == "xmax":
                        xmax = int(ggc.text)
                    if ggc.tag == "ymin":
                        ymin = int(ggc.text)
                    if ggc.tag == "ymax":
                        ymax = int(ggc.text)
        gt.append({name: [xmin,ymin,xmax,ymax]})

    return gt

def ssImg(net, location, imagename):
    img = skimage.io.imread(location + "JPEGImages/" + imagename + ".jpg")
    img_lbl, regions = selectivesearch.selective_search(img, scale=500,
            sigma=0.9, min_size=2000)
    fixed_regions = []
    img_t = transform(img)
    for r in regions:
        x,y,w,h = r['rect']
        toPred = img_t[:,y:y+h,x:x+w]
        toPred = resize(toPred)
        toPred = toPred.unsqueeze(0)
        pred = net(toPred).detach().numpy()
        r.update({'classification': classes[np.argmax(pred)]})
        r.update({'confidence': np.max(pred)})
        del(r['labels'])
        del(r['size'])
        #del(r['rect'])
        r.update({'bbox': torch.tensor([x,y,x+w,y+h])})
        fixed_regions.append(r)
    boxes = torch.zeros([len(fixed_regions),4])
    for i,region in enumerate(fixed_regions):
        boxes[i] = region['bbox']
    confidence = torch.tensor([c['confidence'] for c in fixed_regions])
    nms_indices = nms(boxes,confidence,0.2)
    good_regions = []
    for i,region in enumerate(fixed_regions):
        if i in nms_indices:
            good_regions.append(region)
    
    return good_regions

def IOU(rect1, rect2):
    #print(f"rect1: {rect1}, rect2: {rect2}")
    #print(f"{type(rect1)}, {type(rect2)}")
    # Get intersection
    xleft = max(rect1[0],rect2[0])
    xright = min(rect1[2],rect2[2])
    ytop = max (rect1[1], rect2[1])
    ybottom = min(rect1[3], rect2[3])
    intersection = (xright - xleft) * (ybottom-ytop)
    # Get union
    xleft = min(rect1[0],rect2[0])
    xright = max(rect1[2],rect2[2])
    ytop = min(rect1[1], rect2[1])
    ybottom = max(rect1[3], rect2[3])
    union = (xright - xleft) * (ybottom-ytop)
    
    #overlap = rect1*rect2
    #union = rect1 + rect2
    #print(f"{intersection} / {union} = {intersection/union}")
    return intersection/union

def mAPImg(net, location, imgname):
    gts = getGTs(location + "Annotations/" + imgname + ".xml")
    preds = ssImg(net, location, imgname)
    gt_labels = np.unique([list(gts[i].keys())[0] for i in range(len(gts))])
    precision = np.zeros(len(classes))
    recall = np.zeros(len(classes))
    for gtl in gt_labels:
        count = 0
        index = 0
        for i,c in enumerate(classes):
            if c is gtl:
                index = i
                break
        tp = 0
        fp = 0
        gt_bboxes = []
        numgts = 0
        for gt in gts:
            if list(gt.keys())[0] == gtl:
                gt_bboxes.append(np.asarray(list(gt.values())[0]))
                numgts += 1
        for p in preds:
            if p['classification'] == gtl:
                count += 1
                a,b,c,d = p['rect']
                pred_rect = np.asarray([a,b,a+c,b+d])
                if any(IOU(pred_rect,gt_bboxes[i]) >= 0.5 for i in range(len(gt_bboxes))):
                    tp += 1
        if count > 0:
            precision[index] = tp/count
            recall[index] = tp/numgts
    #print(precision)
    #print(f"Image {imgname} has precision {precision}")
    return precision,recall

class VOCDataset(Dataset):
  '''
  Gives us the dataset for our VOC files
  '''
  def __init__(self,
               datapath,
               classes,
               annotationPath = "Annotations",
               imgDir = "JPEGImages",
               ):
    super(VOCDataset,self).__init__()
    assert(os.path.exists(datapath)),f"Path {datapath} DOES NOT exist"
    if "/" not in datapath[-1]:
      datapath += "/"
    self.datapath = datapath
    if "/" not in annotationPath[-1]:
      annotationPath += "/"
    self.annotationPath = annotationPath
    if "/" not in imgDir[-1]:
      imgDir += "/" 
    self.imgDir = imgDir
    self.data = []
    annotationDir = datapath + annotationPath
    annotations = os.listdir(annotationDir)
    for annotation in annotations:
      tree = ET.parse(annotationDir+annotation)
      root = tree.getroot()
      for child in root:
        if child.tag == "object":
          for gc in child:
            if gc.tag == "name":
              name = gc.text
            if gc.tag == "bndbox":
              for ggc in gc:
                if ggc.tag == "xmin":
                  xmin = int(ggc.text)
                if ggc.tag == "xmax":
                  xmax = int(ggc.text)
                if ggc.tag == "ymin":
                  ymin = int(ggc.text)
                if ggc.tag == "ymax":
                  ymax = int(ggc.text)
          #img = self.cropAndGetImg(datapath + imgDir+annotation[:-4] + ".jpg",
          #                    xmin, xmax, ymin, ymax)
          #self.data.append((img, [xmin,ymin,xmax,ymax], name, annotation[:-4]))
          #index = np.where(np.asarray(classes) == name)[0][0]
          index = 0
          for i in range(len(classes)):
              if classes[i] == name:
                  index = i
                  break
          #label = torch.tensor([index],dtype=torch.long)
          label = torch.zeros(1,dtype=torch.long)
          label[0] = index
          assert(label.size()[0] == 1),\
                  f"Label has size {label.size()} with values {label} in file\
                  {annotation[:-4]} orig {name} {classes}"
          assert(len(label.size()) == 1),\
                  f"Got label with size {label.size()}, {label}, {name}, \
                  {np.where(classes == name)}, {classes}, {index}"
          self.data.append(([xmin,ymin,xmax,ymax],label,annotation[:-4]))
  
  def __len__(self):
    return len(self.data)-1

  def __getitem__(self,idx):
    ''' Returns a tuple (image, [xmin,ymin,xmax,ymax], label name, filenumber)'''
    (minmax, label, filename)= self.data[idx]
    img = self.cropAndGetImg(self.datapath + self.imgDir + filename + ".jpg",
                             minmax[0],minmax[2],minmax[1],minmax[3])
    #print(f"Getting label {label} {label.size()}")
    return (img, minmax, label.item(), filename)
        
  def cropAndGetImg(self,imgStr, xmin, xmax, ymin, ymax):
    img = Image.open(imgStr)
    #print(f"Opened image with size {np.shape(img)}")
    transform = transforms.Compose([
                          transforms.ToTensor(),
                          #transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          #                     std=[0.229, 0.224, 0.225]),
    ])
    resize = transforms.Compose([
                                 transforms.ToPILImage(),
                                 transforms.Resize(256),
                                 #transforms.Resize((64,64)),
                                 transforms.CenterCrop(224),
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                      std=[0.229, 0.224, 0.225]),
    ])
    imgTensor = transform(img)
    c,y,x = imgTensor.size()
    #img = imgTensor[:,xmin:xmax,ymin:ymax]
    img = imgTensor[:,ymin:ymax,xmin:xmax]
    img = resize(img)
    return img

def retrain():
    classes = getClasses(testAnno)
    numClasses = len(classes)
    # Fix net for VOC dataset
    net = models.vgg16(pretrained=True)
    net.classifier[6] = nn.Linear(4096, numClasses)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device {device}")
    epochs = 20 
    batch_size = 64
    #batch_size = 1
    nw = 32
    lr = 0.0001
    dataset = VOCDataset(trainDir,classes)
    print(f"Len dataset = {len(dataset)}")
    data = DataLoader(dataset, 
                      shuffle=True, 
                      batch_size=batch_size, 
                      num_workers=nw)
    net = net.to(device)
    optimizer = optim.AdamW(net.parameters(), lr)
    lossFN = nn.CrossEntropyLoss()
    print(classes)
    classes = np.asarray(classes)
    for epoch in range(epochs):
        running_loss = 0
        for i,(img, _, label, _) in enumerate(data):
            optimizer.zero_grad()
            img = img.to(device)
            label = label.to(device)
            pred = net(img)
            #print(f"Got label size {label.size()} {label} {pred.size()}")
            #print(f"{pred}, {label}, {classes[label]}")
            #toPIL = transforms.ToPILImage()
            #plotme = toPIL(img.squeeze().detach().cpu())
            #print(f"size {np.shape(plotme)}")
            ##plotme = mpimg.imread(plotme)
            #plt.imshow(plotme)
            #plt.savefig("sampleplot")
            #exit(0)

            loss = lossFN(pred,label)
            running_loss += loss.item()
            loss.backward()
            optimizer.step()
        epoch_loss = running_loss/(i+1)
        print(f"Epoch {epoch} finished with loss {epoch_loss}")

    torch.save(net.state_dict(), "retrainedForVOC.pt")


def singleImage(net, location, imagename, classes):
    img = skimage.io.imread(location + "JPEGImages/" + imagename + ".jpg")
    img_lbl, regions = selectivesearch.selective_search(img, scale=500,
            sigma=0.9, min_size=2000)
    fixed_regions = []
    img_t = transform(img)
    for r in regions:
        x,y,w,h = r['rect']
        toPred = img_t[:,y:y+h,x:x+w]
        toPred = resize(toPred)
        toPred = toPred.unsqueeze(0)
        pred = net(toPred).detach().numpy()
        r.update({'classification': classes[np.argmax(pred)]})
        r.update({'confidence': np.max(pred)})
        del(r['labels'])
        del(r['size'])
        #del(r['rect'])
        r.update({'bbox': torch.tensor([x,y,x+w,y+h])})
        fixed_regions.append(r)

    gt = []
    tree = ET.parse(location + "Annotations/" + imagename + ".xml")
    root = tree.getroot()
    for child in root.findall('object'):
        name = child.find('name').text
        for gc in child:
            if gc.tag == "bndbox":
                for ggc in gc:
                    if ggc.tag == "xmin":
                        xmin = int(ggc.text)
                    if ggc.tag == "xmax":
                        xmax = int(ggc.text)
                    if ggc.tag == "ymin":
                        ymin = int(ggc.text)
                    if ggc.tag == "ymax":
                        ymax = int(ggc.text)
        gt.append({name: [xmin,ymin,xmax,ymax]})


    boxes = torch.zeros([len(fixed_regions),4])
    for i,region in enumerate(fixed_regions):
        boxes[i] = region['bbox']
    confidence = torch.tensor([c['confidence'] for c in fixed_regions])
    nms_indices = nms(boxes,confidence,0.2)

    print(len(nms_indices))
    fig, ax = plt.subplots(ncols=1,nrows=1, figsize=(6,6))
    ax.imshow(img)
    for i,c in enumerate(fixed_regions):
        if i not in nms_indices: continue
        x,y,w,h = c['rect']
        classify = c['classification']
        con = "{:0.2f}".format(c['confidence'])
        rect = mpatches.Rectangle((x,y),
                                  w,h, fill=False, edgecolor='red',
                                  linewidth=2)
        ax.add_patch(rect)
        ax.text(x+w/2, y, 
                color='white',
                s=classify + " " + con,
                horizontalalignment='center',
                bbox=dict(facecolor='black'))
    for i,c in enumerate(gt):
        name = list(c.keys())[0]
        xmin,ymin,xmax,ymax = list(c.values())[0]
        rect = mpatches.Rectangle((xmin,ymin),
                                  xmax - xmin,
                                  ymax - ymin,
                                  fill=False,
                                  edgecolor="green",
                                  linewidth=2)
        ax.add_patch(rect)
        ax.text(xmin + (xmax-xmin)/2, ymin,
                color='white',
                s = name + "(GT)",
                horizontalalignment='center',
                bbox=dict(facecolor='black'))
    plt.savefig("example.png")

def allImgs(net):
    annotations = os.listdir(testAnno)
    precision = []
    recall = []
    for i,anno in enumerate(annotations):
        prec,rec = mAPImg(net, testDir, anno[:-4])
        precision.append(prec)
        recall.append(rec)
        print(f"[{i}/{len(annotations)}]: R = {rec.mean()}, AP = {prec.mean()}")
    accum_prec = np.asarray(precision).mean(axis=0)
    accum_rec  = np.asarray(recall).mean(axis=0)
    print(f"Name\tRecall\tAP")
    for i in range(len(classes)):
        print(f"{classes[i]}\t{accum_rec[i]}\t{accum_prec[i]}")
    mAP = accum_prec.mean()
    print(f"mAP\t\t\t{mAP}")

def main():
    if not os.path.isfile('retrainedForVOC.pt'):
        retrain()
    classes = getClasses(testAnno)
    numClasses = len(classes)
    # Fix net for VOC dataset
    net = models.vgg16(pretrained=True)
    net.classifier[6] = nn.Linear(4096, numClasses)
    net.load_state_dict(torch.load('retrainedForVOC.pt'))
    net.eval()

    #img = skimage.io.imread(testDir + "JPEGImages/000718.jpg")
    #singleImage(net, testDir + "JPEGImages/000718.jpg", classes)
    #singleImage(net, testDir, "008545", classes)
    #mAPImg(net, testDir, "008545")
    allImgs(net)

if __name__ == '__main__':
    main()
    #retrain()
