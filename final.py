from ultralytics import YOLO
import cv2
from transformers import TrOCRProcessor
from transformers import VisionEncoderDecoderModel
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
import time
import argparse
import json

yolo_model=YOLO("best.onnx")
ocr_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-large-stage1")
ocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-large-stage1")
tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
bert_model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")

def get_bounding_boxes_YOLO(img):
    results = yolo_model.predict(img, imgsz=1280)
    result = results[0]
    final_boxes = []
    for box in result.boxes:
        cords = box.xyxy[0].tolist()
        cords = [round(x) for x in cords]
        conf = round(box.conf[0].item(), 2)
        if conf > 0.5:
            final_boxes.append(cords)

            # merge overlapping boxes
    merged_boxes = []

    for box in final_boxes:
        box_merged = False
        for i, merged_box in enumerate(merged_boxes):
            if (
                    box[0] >= merged_box[0] and box[1] >= merged_box[1] and
                    box[2] <= merged_box[2] and box[3] <= merged_box[3]
            ):
                # box is completely inside an existing merged box
                box_merged = True
                break
            elif (
                    box[0] <= merged_box[2] and box[1] <= merged_box[3] and
                    box[2] >= merged_box[0] and box[3] >= merged_box[1]
            ):
                # box is overlapping with an existing merged box
                merged_boxes[i] = [
                    min(box[0], merged_box[0]),
                    min(box[1], merged_box[1]),
                    max(box[2], merged_box[2]),
                    max(box[3], merged_box[3])
                ]
                box_merged = True
                break

        if not box_merged:
            merged_boxes.append(box)
    return merged_boxes

def crop_image_according_to_boxes(img, boxes):
    cropped_images = []
    for box in boxes:
        cropped_images.append(img[box[1]:box[3], box[0]:box[2]])
    return cropped_images

def split_image(image):
    image1 = image[:image.shape[0]//2, :]
    image2 = image[image.shape[0]//2:, :]
    return image1, image2


parser = argparse.ArgumentParser(description='OCR on image')
parser.add_argument('--image_path', type=str, required=True, help='Path to the input image')
args = parser.parse_args()
path = args.image_path

start = time.time()
img = cv2.imread(path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# get left half
img = img[:, :img.shape[1]//2]

boxes = get_bounding_boxes_YOLO(img)
images = crop_image_according_to_boxes(img, boxes)
yolo_time = time.time() - start
start = time.time()
final_images = []
for image in images:
    image1, image2 = split_image(image)
    final_images.append(image1)
    final_images.append(image2)

j=0
names=[]
positions=[]
while j < len(final_images):
    texts = []
    for image in final_images[j:j+2]:
        pixel_values = ocr_processor(image, return_tensors="pt").pixel_values
        generated_ids = ocr_model.generate(pixel_values)
        generated_text = ocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        texts.append(generated_text)

    for i in range(len(texts)):
        texts[i] = texts[i].split()

    for i in range(len(texts)):
        texts[i] = [x.title() for x in texts[i]]

    nlp = pipeline("ner", model=bert_model, tokenizer=tokenizer)

    name = []
    name_found = False
    for text in texts:
        if name_found:
            break
        for t in text:
            ner_result = nlp(t)
            for i in range(len(ner_result)):
                if ner_result[i]["entity"] == "B-PER" or ner_result[i]["entity"] == "I-PER":
                    name = t
                    name_found = True
                    break
            if name_found:
                break

    if not name_found:
        name = "Unknown"

    possible_poker_positions = ["btn", "utg", "bb", "hj", "cut", "sb", "3b", "utgis", "+1", "+2"]
    position_found = False
    texts = [item for sublist in texts for item in sublist]
    for t in texts:
        if t.lower() in possible_poker_positions:
            position = t.upper()
            position_found = True
            break

    if not position_found:
        position = "Unknown"

    names.append(name)
    positions.append(position)

    j+=2

ocr_end = time.time() - start
result_list = []
result_dict = {}
for i in range(len(names)):
    result_dict = {"name": names[i], "position": positions[i]}
    result_list.append({i: result_dict})
json_result = json.dumps(result_list, indent=2)
with open("result.json", "w") as f:
    f.write(json_result)
print("\n\nResults saved to results.json.")
print("Time taken for YOLO: ", yolo_time)
print("Time taken for OCR and NER: ", ocr_end)
print("Total time taken: ", yolo_time + ocr_end)







