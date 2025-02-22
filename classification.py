import cv2
import math

DESIRED_HEIGHT = 480
DESIRED_WIDTH = 480

def resize_and_show(image):
  h, w = image.shape[:2]
  if h < w:
    img = cv2.resize(image, (DESIRED_WIDTH, math.floor(h/(w/DESIRED_WIDTH))))
  else:
    img = cv2.resize(image, (math.floor(w/(h/DESIRED_HEIGHT)), DESIRED_HEIGHT))
#   cv2_imshow(img)
  cv2.imshow("test",img)
  cv2.waitKey(0)

IMAGE_FILENAMES = ['images/burger.jpg', 'images/cat.jpg']

# Preview the images.

images = {name: cv2.imread(name) for name in IMAGE_FILENAMES}
for name, image in images.items():
  print(name)
  resize_and_show(image)




  # STEP 1: Import the necessary modules.
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python.components import processors
from mediapipe.tasks.python import vision

# STEP 2: Create an ImageClassifier object.
base_options = python.BaseOptions(model_asset_path='models\efficientnet_lite2.tflite')
options = vision.ImageClassifierOptions(
    base_options=base_options, max_results=8)
classifier = vision.ImageClassifier.create_from_options(options)

#   # STEP 3: Load the input image.
image = mp.Image.create_from_file(IMAGE_FILENAMES[0])

#   # STEP 4: Classify the input image.
classification_result = classifier.classify(image)

  # STEP 5: Process the classification result. In this case, visualize it.
top_category = classification_result.classifications[0].categories[0]
print(top_category)

results = classification_result.classifications[0].categories
print(results)

# images = []
# predictions = []
# for image_name in IMAGE_FILENAMES:
#   # STEP 3: Load the input image.
#   image = mp.Image.create_from_file(image_name)

#   # STEP 4: Classify the input image.
#   classification_result = classifier.classify(image)

#   # STEP 5: Process the classification result. In this case, visualize it.
#   images.append(image)
#   top_category = classification_result.classifications[0].categories[0]
#   predictions.append(f"{top_category.category_name} ({top_category.score:.2f})")

# display_batch_of_images(images, predictions)