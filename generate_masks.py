import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import requests
from ultralytics import SAM
import cv2
import numpy as np

class GroundingDINO_wrapper:

    def __init__(self, model_name:str, device:str):
      self.__model_id = f"IDEA-Research/{model_name}"
      self.__device = device
      self.__model = AutoModelForZeroShotObjectDetection.from_pretrained(
          self.__model_id).to(self.__device)

    def predict_single_image(self, text_labels:list[str], image, threshold=0.4, text_threshold=0.3):
      if isinstance(image, str):
        if image.startswith("http"):
          image = Image.open(requests.get(image, stream=True).raw)
        else:
          image = Image.open(image)
      processor = AutoProcessor.from_pretrained(self.__model_id)
      inputs = processor(images=image, text=text_labels, return_tensors="pt").to(self.__model.device)
      with torch.no_grad():
        outputs = self.__model(**inputs)
      return processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        threshold=threshold,
        text_threshold=text_threshold,
        target_sizes=[image.size[::-1]]
      )[0]

    def move_to_device(self,new_device:str) -> None:
      self.__model.to(new_device)
      self.__device = new_device


class SAM_wrapper:
    def __init__(self, model_name:str, device:str):
      self.__model = SAM(f"{model_name}.pt")
      self.__device = device
      self.__model = self.__model.to(self.__device)
    def predict_single_image(self, image:str, bboxes:list[list[float]], labels:list[str]):
      result = self.__model([image],bboxes=bboxes)[0]
      result.names = {i:labels[i] for i in range(len(labels))}
      return result


class Grounded_SAM:
    def __init__(self, grounding_dino_model_name, sam_model_name, device:str):
      self.grounding_dino_model = GroundingDINO_wrapper(grounding_dino_model_name,device)
      self.sam_model = SAM_wrapper(sam_model_name,device)
    def predict_single_image(self, image:str, text_labels:list[str], threshold=0.4, text_threshold=0.3):
      detection_results = self.grounding_dino_model.predict_single_image(text_labels,image,threshold,text_threshold)
      if len(detection_results["boxes"]) != 0:
        return self.sam_model.predict_single_image(image,detection_results["boxes"].detach().cpu().tolist(),detection_results["labels"])
      return None

names_to_idxs = {
    "wall": 0, # not in the prompt, everything not detected is classified as wall
    "door": 10,
    "basket": 20,
    "box": 20,
    "floor": 30,
    "human": 40,
    "forklift": 50,
    "rack": 60,
    "wood": 70,
    "pallete": 70, 
    "container": 80
}

prompt = ["door","basket","box","floor","human","forklift","rack","wood","pallete", "container"]
# prompt = ["object", "floor", "human"]

# idxs to colors

model = Grounded_SAM("grounding-dino-tiny","sam2.1_l","cuda")

def produce(src, dst, model, prompt, save_vis=False):
    result = model.predict_single_image(src,prompt)
    if result is None:
       return
    if save_vis:
        result.save(dst+" visualization.png")
    masks = np.zeros((cv2.imread(src).shape[0],cv2.imread(src).shape[1]))
    for idx in result.names.keys():
        pname = result.names[idx].split(" ")[0]
        try:
            mask = np.asarray(result.masks.data[idx].detach().cpu()).astype(int) * names_to_idxs[pname]
        except KeyError:
            continue
        masks = masks + mask * (masks == 0).astype(int) # ignore overlapped instances
    cv2.imwrite(dst, masks)
    print(f"Success: {dst}")

# import os
# path = "/var/local/storage/kfoteinos/tornado_subset_2000"
# for file in os.listdir(path+"/rgb"):
#     if not file.endswith(".png") or not file.startswith("rgb_"):
#        print("WARNING: "+file)
#        continue
#     idx = int(file.split("_")[1].split(".")[0])
#     print(f"{file} -> {idx}")
#     produce(os.path.join(path, "rgb", file), os.path.join(path, "semantic_class", f"semantic_class_{idx}.png"), model, prompt)

# os.system(f"cp -r {path} /home/kfoteinos/VSLAM/tornado_subset_2000")

for i in range(100):
    file = f"/home/kfoteinos/VSLAM/tornado_subset/rgb/rgb_{i}.png"
    print(file)
    produce(file, f"/home/kfoteinos/VSLAM/tornado_subset/semantic_class/semantic_class_{i}.png", model, prompt)