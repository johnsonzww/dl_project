# Obstructions removal from images based on layer decomposition and reconstruction

Yixuan Lyu, Wenwei Zhang

## Introduction

Obstruction in images would cause inconvenience in many scenarios. Since taking photo has been easier with smart phones, taking photo through various of obstructions has become a common issue. For instance, when we are trying to take a large format photograph by using the wide-angle lens on our smart phone, many obstructions will show up as an unintended guest as more content can be captured by a single shot. When we want to look through glasses, reflection of light often cause inconvenience for our perception. Though these kind of problem can be relieved by physical method like adding anti-reflection coating, it also cost more and wastes human-power. Deep learning model can help us with such problem. With pre-trained model, clean and clear background can be separated or regenerate from the mixed figure. In our project, we tend to use 5 frames of obstruction image as input, clean background and obstruction images as output.

## Requirement

- TensorFlow 1.x

## Usage

### access via code 
1. edit config file *run_fence.py*
   - TRAINING_DATA_PATH: where test data is stored.
   - TRAINING_SCENE: the scene number, your test case should be name as *"TRAINING_SCENE_I0"* - *"TRAINING_SCENE_I4"*, which is a 5 frame sequence of picture. *e.g. TRAINING_SCENE="00001", pics: 00001_I0.png, 00001_I1.png, 00001_I2.png, 00001_I3.png, 00001_I4.png*.
   - OUTPUT_DIR: dir that the result is stored.
   - OPTIMIZATION_STEPS: how many epochs to be trained in online training.  
2. prepared a 5 frame sequence and put it in to *TRAINING_DATA_PATH*
3. run the program with:
   - ``` python run_fence.py ```

### access via colab
https://colab.research.google.com/drive/1phexVRtCPbVJxfj0cjhBrT6nlvhEljGU?usp=sharing