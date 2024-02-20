import React, { Component } from "react";
import ReactDOM from "react-dom";
import MagicDropzone from "react-magic-dropzone";

import "./styles.css";
const tf = require('@tensorflow/tfjs');

window.addEventListener('unhandledrejection', event => {
  console.error('Unhandled Promise Rejection:', event.reason);
});

const detectionModelWeights = 'web_model1/model.json';
const segmentationModelWeights = 'web_model2/model.json';

const names = ['mass']

class App extends Component {
  state = {
    detectionModel: null,
    segmentationModel: null,
    preview: "",
    detectionResults: []
  };

  // Declare ctx as a class property
  ctx = null;

  async componentDidMount() {
    try {
      const detectionModel = await tf.loadGraphModel(detectionModelWeights);
      const segmentationModel = await tf.loadGraphModel(segmentationModelWeights);

      this.setState({
        detectionModel,
        segmentationModel,
      });
      console.log('Models loaded successfully:', detectionModel, segmentationModel);
    } catch (error) {
      console.error('Error loading models:', error);
      // Handle the error, e.g., display an error message to the user
    }
  }

  onDrop = (accepted, rejected, links) => {
    this.setState({ preview: accepted[0].preview || links[0] });
  };

  cropToCanvas = (image, canvas) => {
    const naturalWidth = image.naturalWidth;
    const naturalHeight = image.naturalHeight;

    this.ctx = canvas.getContext("2d");

    this.ctx.clearRect(0, 0, this.ctx.canvas.width, this.ctx.canvas.height);
    this.ctx.fillStyle = "#000000";
    this.ctx.fillRect(0, 0, canvas.width, canvas.height);

    const ratio = Math.min(canvas.width / naturalWidth, canvas.height / naturalHeight);
    const newWidth = Math.round(naturalWidth * ratio);
    const newHeight = Math.round(naturalHeight * ratio);

    this.ctx.drawImage(
      image,
      0,
      0,
      naturalWidth,
      naturalHeight,
      (canvas.width - newWidth) / 2,
      (canvas.height - newHeight) / 2,
      newWidth,
      newHeight,
    );
  };

  onImageChange = async (e) => {
    const c = document.getElementById("canvas");
    this.cropToCanvas(e.target, c);
    let [modelWidth, modelHeight] = this.state.detectionModel.inputs[0].shape.slice(1, 3);
    const input = tf.tidy(() => {
      return tf.image.resizeBilinear(tf.browser.fromPixels(c), [modelWidth, modelHeight])
        .div(255.0).expandDims(0);
    });

    // Perform object detection
    const detectionResults = await this.detectObjects(input, c);

    this.setState({
      detectionResults
    });
  };

  async detectObjects(input, c) {

    const ctx = this.ctx;

    const res = await this.state.detectionModel.executeAsync(input);

    // Font options.
    const font = "16px sans-serif";
    ctx.font = font;
    ctx.textBaseline = "top";

    const [boxes, scores, classes, valid_detections] = res;
    const boxes_data = boxes.dataSync();
    const scores_data = scores.dataSync();
    const classes_data = classes.dataSync();
    const valid_detections_data = valid_detections.dataSync()[0];

    tf.dispose(res)

    const detectionResults = [];

    var i;
    for (i = 0; i < valid_detections_data; ++i){
      let [x1, y1, x2, y2] = boxes_data.slice(i * 4, (i + 1) * 4);

      x1 *= c.width;
      x2 *= c.width;
      y1 *= c.height;
      y2 *= c.height;

      const width = Math.abs(x2 - x1);
      const height = Math.abs(y2 - y1);

      const klass = names[classes_data[i]];
      const score = scores_data[i].toFixed(2);

      const extraSpace = 10; // Adjust this value based on how much extra space you want

      // Assuming x1, x2, y1, y2 are your bounding box coordinates
      // and width, height are the dimensions of the image

      // Add extra space to the bounding box coordinates
      const expandedX1 = Math.max(0, x1 - extraSpace);
      const expandedX2 = Math.min(640, x2 + extraSpace);
      const expandedY1 = Math.max(0, y1 - extraSpace);
      const expandedY2 = Math.min(640, y2 + extraSpace);

      // Update the bounding box coordinates in your detectionResults array
      detectionResults.push({
        class: klass,
        score: parseFloat(score), // Convert score to float
        box: {
          x1: expandedX1,
          y1: expandedY1,
          x2: expandedX2,
          y2: expandedY2,
          width,
          height,
        },
      });

      console.log('i', i);

      const segmentationResults = await this.segmentDetectedArea(c, detectionResults);

      await this.visualizeSegmentationMask(c, segmentationResults, {
      x1: x1,
      y1: y1,
      x2: x2,
      y2: y2,
      width,
      height,
        });

      // Move the bounding box 10 pixels (adjust as needed) upwards
      const yOffset = 10; // Adjust as needed
      y1 -= yOffset;
      y2 -= yOffset;

      // Calculate the new width and height
      const newWidth = width * 1.2; // Increase the width by 20% or adjust as needed
      const newHeight = height * 1.2; // Increase the height by 20% or adjust as needed

      // Calculate the adjustment in x and y to keep the center the same
      const dx = (newWidth - width) / 2;
      const dy = (newHeight - height) / 2;

      // Adjust the coordinates to keep the center the same
      x1 -= dx;
      y1 -= dy;
      x2 += dx;
      y2 += dy;

      // Draw the bounding box.
      ctx.strokeStyle = "#00FFFF";
      ctx.lineWidth = 2;
      ctx.strokeRect(x1, y1, width+10, height+10);

      // Draw the label background.
      ctx.fillStyle = "#00FFFF";
      const fontSize = 12; // Adjust the font size as needed
      ctx.font = `${fontSize}px sans-serif`;
      const textWidth = ctx.measureText(klass + ":" + score).width;
      const textHeight = fontSize; // Use the font size for text height
      ctx.fillRect(x1, y1, textWidth + 4, textHeight + 4);

    }

    for (i = 0; i < valid_detections_data; ++i){
        let [x1, y1, , ] = boxes_data.slice(i * 4, (i + 1) * 4);
        x1 *= c.width;
        y1 *= c.height;
        const klass = names[classes_data[i]];
        const score = scores_data[i].toFixed(2);

        // Draw the text last to ensure it's on top.
        ctx.fillStyle = "#000000";
        ctx.fillText(klass + ":" + score, x1 - 5, y1 - 10);

      }

    return detectionResults;
  }

  async visualizeSegmentationMask(c, segmentationResults, box) {

    // Assuming segmentationResult is a TensorFlow.js tensor
    const segmentationResultArray = await segmentationResults.data();

    // Assuming width and height are the dimensions of the segmentation result
    const maskWidth = 128;
    const maskHeight = 128;

    // Set a threshold value to determine object vs. background
    const threshold = 0.5;

    const ctx = c.getContext('2d');

    // Calculate the scaling factors for resizing the mask
    const scaleX = box.width / maskWidth;
    const scaleY = box.height / maskHeight;

    // Visualize the binary segmentation mask
    for (let i = 0; i < maskHeight; i++) {
      for (let j = 0; j < maskWidth; j++) {
        const maskValue = segmentationResultArray[i * maskWidth + j];
        const binaryValue = maskValue > threshold ? 1 : 0;
        const color = binaryValue === 1 ? 'rgba(255, 255, 255, 0.5)' : 'rgba(0, 0, 0, 0)';

        // Draw a pixel on the canvas at the correct position within the resized bounding box
        ctx.fillStyle = color;
        ctx.fillRect(box.x1 + Math.round(j * scaleX), box.y1 + Math.round(i * scaleY), 1, 1);
      }
    }

    // Create a canvas element
    const canvas = document.createElement('canvas');
    canvas.width = 128;
    canvas.height = 128;
    document.body.appendChild(canvas); // Append the canvas to the body or another container

    // Get 2D context of the canvas
    const ctx1 = canvas.getContext('2d');

    // Visualize the binary segmentation mask
    for (let i = 0; i < canvas.height; i++) {
      for (let j = 0; j < canvas.width; j++) {
        const maskValue = segmentationResultArray[i * canvas.width + j];

        // Use a threshold to decide object vs. background
        const binaryValue = maskValue > threshold ? 1 : 0;

        // Assuming you want to use white for the object and black for the background
        const color = binaryValue === 1 ? 'rgba(255, 255, 255, 1)' : 'rgba(0, 0, 0, 1)';

        // Draw a pixel on the canvas
        ctx1.fillStyle = color;
        ctx1.fillRect(j, i, 1, 1);
      }
    }
  }

  async segmentDetectedArea(c, detectionResults) {

    // Check if detectionResults is undefined or an empty array
    if (!detectionResults || detectionResults.length === 0) {
      console.error('No detection results or empty array:', detectionResults);
      return null; // Handle the error appropriately
    }

    // Take the first detection result
    const firstDetectionResult = detectionResults[0];

    const { box } = firstDetectionResult;

    // Check if box is undefined or doesn't have the expected structure
    if (!box || typeof box !== 'object' || Object.keys(box).length < 4) {
      console.error('Invalid box structure:', box);
    }

    const { x1, y1, x2, y2 } = box

    // Crop the detected area
    const croppedTensor = tf.tidy(() => {
      const imageTensor = tf.browser.fromPixels(c);

      console.log('Original ROI coordinates and dimensions:', x1, y1);

      const expandedImage = imageTensor.expandDims();

      // Assuming x1, y1, x2, y2 are your bounding box coordinates
      const bbox = [y1/640, x1/640, y2/640, x2/640];

      console.log('Normalized ROI coordinates and dimensions:',bbox);

      // Convert the bounding box coordinates to TensorFlow.js format
      const boxes = tf.tensor2d([bbox], [1, 4]); // Provide the shape [1, 4] to indicate a batch size of 1

      // Extract ROI using tf.image.cropAndResize
      const roiTensor = tf.image.cropAndResize(
        expandedImage,
        boxes,
        [0],  // Indices of the boxes, assuming a batch size of 1
        [128, 128]  // Specify the desired size of the output ROI
      );

      // Remove the batch dimension
      const finalRoiTensor = roiTensor.squeeze();

      // Get the dimensions of the tensor
      const [height, width, channels] = finalRoiTensor.shape;

      // Convert the tensor to a TypedArray
      const flattenedData = finalRoiTensor.dataSync();

      // Create a TensorFlow.js Tensor from the flattened data
      const imageTensor2 = tf.tensor(flattenedData, [height, width, channels]).div(255.0);

      return imageTensor2;
    });

    // Resize the cropped image to [128, 128]
    const resizedTensor = tf.image.resizeBilinear(croppedTensor, [128, 128]);

    const expandTensor = resizedTensor.expandDims(0)

    const finalTensor = expandTensor.transpose([0, 3, 2, 1])

    // Perform segmentation
    const segmentationResult = await this.state.segmentationModel.executeAsync(finalTensor);

    console.log('segmentationResult:', await segmentationResult.data());

    const width = x2-x1
    const height = y2-y1

    console.log('width, height:', width, height);

    // Resize the segmentationResult tensor to match the bounding box dimensions
    const resizedSegmentationResult = tf.image.resizeBilinear(segmentationResult, [width, width]);

     // Dispose of the tensors
    croppedTensor.dispose();
    
    return segmentationResult;
}   

  render() {
    return (
      <div className="Dropzone-page">
        {this.state.detectionModel ? (
          <MagicDropzone
            className="Dropzone"
            accept="image/jpeg, image/png, .jpg, .jpeg, .png"
            multiple={false}
            onDrop={this.onDrop}
          >
            {this.state.preview ? (
              <img
                alt="upload preview"
                onLoad={this.onImageChange}
                className="Dropzone-img"
                src={this.state.preview}
              />
            ) : (
              "Choose or drop a file."
            )}
            <canvas id="canvas" width="640" height="640" />
          </MagicDropzone>
        ) : (
          <div className="Dropzone">Loading model...</div>
        )}
      </div>
    );
  }
}


const rootElement = document.getElementById("root");
ReactDOM.render(<App />, rootElement);