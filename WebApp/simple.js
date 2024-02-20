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

    // Perform segmentation on the detected area
    const segmentationResults = await this.segmentDetectedArea(c, detectionResults);

    this.setState({
      detectionResults,
      segmentationResults,
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
      console.log('Original values:', x1, y1, x2, y2);

      x1 *= c.width;
      x2 *= c.width;
      y1 *= c.height;
      y2 *= c.height;

      console.log('Scaled values:', x1, y1, x2, y2);

      const width = Math.abs(x2 - x1);
      const height = Math.abs(y2 - y1);

      console.log('Calculated width and height:', width, height);
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

      console.log('Scaled values:', expandedX1, expandedY1, expandedX2, expandedY2);

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

      if (segmentationResults) {
      this.drawSegmentationMask(ctx, segmentationResults, x1, y1, 128, 128);
    }


      // Draw the bounding box.
      ctx.strokeStyle = "#00FFFF";
      ctx.lineWidth = 4;
      ctx.strokeRect(x1, y1, width, height);

      // Draw the label background.
      ctx.fillStyle = "#00FFFF";
      const textWidth = ctx.measureText(klass + ":" + score).width;
      const textHeight = parseInt(font, 10); // base 10
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
      ctx.fillText(klass + ":" + score, x1, y1);

    }

    return detectionResults;
  }

  drawSegmentationMask(ctx, segmentationResult, x, y, width, height) {
  const maskData = segmentationResult.dataSync();
  const maskWidth = segmentationResult.shape[2];
  const maskHeight = segmentationResult.shape[1];

  const scaleX = width / maskWidth;
  const scaleY = height / maskHeight;

  for (let i = 0; i < maskHeight; i++) {
    for (let j = 0; j < maskWidth; j++) {
      const maskValue = maskData[i * maskWidth + j];
      const pixelX = Math.floor(x + j * scaleX);
      const pixelY = Math.floor(y + i * scaleY);

      // Adjust the alpha channel based on the segmentation mask value
      ctx.fillStyle = `rgba(0, 255, 0, ${maskValue})`;
      ctx.fillRect(pixelX, pixelY, scaleX, scaleY);
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

    // Log the structure of the firstDetectionResult for debugging
    console.log('firstDetectionResult:', firstDetectionResult);

    const { class: className, score, box } = firstDetectionResult;

    // Check if box is undefined or doesn't have the expected structure
    if (!box || typeof box !== 'object' || Object.keys(box).length < 4) {
      console.error('Invalid box structure:', box);
      return null; // Handle the error appropriately
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

      // Visualize the cropped image during development
      tf.browser.toPixels(imageTensor2).then(pixels => {
        const imageData = new ImageData(new Uint8ClampedArray(pixels), width, height);
        const canvasId = 'visualizationCanvas';

        let canvas = document.getElementById(canvasId);

        if (!canvas) {
          canvas = document.createElement('canvas');
          canvas.id = canvasId;
          document.body.appendChild(canvas);
        }

        canvas.width = width;
        canvas.height = height;
        const context = canvas.getContext('2d');
        context.putImageData(imageData, 0, 0);
      });

      return finalRoiTensor;
    });


    // Resize the cropped image to [128, 128]
    const resizedTensor = tf.image.resizeBilinear(croppedTensor, [128, 128]);

    // Perform segmentation
    const segmentationResult = await this.state.segmentationModel.executeAsync(resizedTensor.transpose([2, 0, 1]).expandDims(0));

    console.log('segmentationResult:', await segmentationResult.data());

    // Dispose of the tensors
    croppedTensor.dispose();
    resizedTensor.dispose();
    
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