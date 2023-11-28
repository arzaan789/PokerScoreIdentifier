This is my submission to the technical assessment for BleedAI.
## Problem Statement
Extract player names and player positions from a given image of a Texas Hold'em poker game.
## Approach and Methodology
First it is needed to extract the location of where this text can be found in the image so the first part was a text detection task. <br>
Obviously a naive approach would be to just split the image into many horizontal lines but that would not be sustainable. <br>
So I used the EAST text detector to extract the location of the text in the image. However, it wasn't that good at detecting all the required text so I used YOLOv8. <br>
After that it was just a matter of sending the localized portions of the image to trOCR and extracting the names using Named Entity Recognition and positions by a simple search.
## Details of my experimentation and exploration
A detailed presentation in a notebook format can be found in the notebook ```Exploration and Experimentation.ipynb```. It will take you each step of the way from the initial text detection to the final result and
all the ideas I had in between (including any references). <br>
All my thought processes and ideas are documented in the notebook and it is a detailed working of how I approached the problem.
## How to run the code
1. Clone the repository
2. Run the following command to install the dependencies
```pip install -r requirements.txt```
3. Run the following command to run the code
```python final.py --image_path "path to image"``` where ```path to image``` is the path to the image you want to extract the names and positions from.
4. The output will be saved in a JSON file called ```result.json```.
5. There will also be a terminal output of the time it took to run the extraction.

## Note
Python version used: 3.11 <br>
Hours spent on the task: Approximately 14 hours