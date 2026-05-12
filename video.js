document.getElementById("ai").addEventListener("change", toggleAi)
document.getElementById("fps").addEventListener("input", changeFps)

const video = document.getElementById("video");
const c1 = document.getElementById('c1');
const ctx1 = c1.getContext('2d');
var cameraAvailable = false;
var aiEnabled = false;
var fps = 16;
var confidenceThreshold = 0.5;

/* Setting up the constraint */
var facingMode = "environment"; // Can be 'user' or 'environment' to access back or front camera (NEAT!)
var constraints = {
    audio: false,
    video: {
        facingMode: facingMode
    }
};

/* Stream it to video element */
camera();
function camera() {
    if (!cameraAvailable) {
        console.log("camera")
        navigator.mediaDevices.getUserMedia(constraints).then(function (stream) {
            cameraAvailable = true;
            video.srcObject = stream;
        }).catch(function (err) {
            cameraAvailable = false;
            if (modelIsLoaded) {
                if (err.name === "NotAllowedError") {
                    document.getElementById("loadingText").innerText = "Waiting for camera permission";
                }
            }
            setTimeout(camera, 1000);
        });
    }
}

window.onload = function () {
    timerCallback();
}

function timerCallback() {
    if (isReady()) {
        setResolution();
        ctx1.drawImage(video, 0, 0, c1.width, c1.height);
        if (aiEnabled) {
            ai();
        }
    }
    setTimeout(timerCallback, fps);
}

function isReady() {
    if (modelIsLoaded && cameraAvailable) {
        document.getElementById("loadingText").style.display = "none";
        document.getElementById("ai").disabled = false;
        return true;
    } else {
        return false;
    }
}

function setResolution() {
    if (window.screen.width < video.videoWidth) {
        c1.width = window.screen.width * 0.9;
        let factor = c1.width / video.videoWidth;
        c1.height = video.videoHeight * factor;
    } else if (window.screen.height < video.videoHeight) {
        c1.height = window.screen.height * 0.50;
        let factor = c1.height / video.videoHeight;
        c1.width = video.videoWidth * factor;
    }
    else {
        c1.width = video.videoWidth;
        c1.height = video.videoHeight;
    }
};

function toggleAi() {
    aiEnabled = document.getElementById("ai").checked;
}

function changeFps() {
    fps = 1000 / document.getElementById("fps").value;
}

function changeConfidence() {
    confidenceThreshold = parseFloat(document.getElementById("confidence").value);
}

async function ai() {
    const confidence = parseFloat(document.getElementById("confidence").value);
    
    if (currentModel === 'yolov8' && yolov8Session) {
        // Use YOLOv8 detection
        try {
            await detectYolov8(c1, confidence);
        } catch (err) {
            console.error("YOLOv8 detection error:", err);
        }
    } else if (currentModel === 'yolo-coco' && yoloDetector) {
        // Use YOLO detection
        try {
            const predictions = await yoloDetector.estimateObjects(c1, confidence);
            drawDetections(predictions, 'yolo');
        } catch (err) {
            console.error("YOLO detection error:", err);
        }
    } else {
        // Use COCO-SSD detection (ml5)
        objectDetector.detect(c1, (err, results) => {
            if (err) {
                console.error(err);
                return;
            }
            // Filter by confidence
            const filtered = results.filter(r => r.confidence >= confidence);
            drawDetections(filtered, 'cocossd');
        });
    }
}

async function detectYolov8(canvas, confidence) {
    try {
        // Prepare input - YOLOv8 expects 640x640 input
        const inputSize = 640;
        const tempCanvas = document.createElement('canvas');
        tempCanvas.width = inputSize;
        tempCanvas.height = inputSize;
        const tempCtx = tempCanvas.getContext('2d');
        
        // Draw and scale image to 640x640
        tempCtx.drawImage(canvas, 0, 0, inputSize, inputSize);
        
        // Convert to ImageData and normalize
        const imageData = tempCtx.getImageData(0, 0, inputSize, inputSize);
        const data = imageData.data;
        
        // Create tensor - YOLOv8 expects RGB normalized to 0-1
        const rgbArray = [];
        for (let i = 0; i < data.length; i += 4) {
            rgbArray.push(data[i] / 255);      // R
            rgbArray.push(data[i + 1] / 255);  // G
            rgbArray.push(data[i + 2] / 255);  // B
        }
        
        // Create input tensor
        const inputTensor = new ort.Tensor('float32', rgbArray, [1, 3, inputSize, inputSize]);
        
        // Run inference
        const feeds = { 'images': inputTensor };
        const results = await yolov8Session.run(feeds);
        
        // Parse results
        const output = results.output0.data;
        const outputShape = results.output0.dims;
        
        // YOLOv8 output format: [1, 84, 8400]
        // 84 = 4 (bbox) + 80 (classes)
        const predictions = [];
        const stride = 8400;
        
        for (let i = 0; i < stride; i++) {
            // Get class probabilities
            let maxConfidence = 0;
            let classId = 0;
            
            for (let j = 0; j < 80; j++) {
                const classConfidence = output[4 * stride + j * stride + i];
                if (classConfidence > maxConfidence) {
                    maxConfidence = classConfidence;
                    classId = j;
                }
            }
            
            // Filter by confidence threshold
            if (maxConfidence >= confidence) {
                // Get bounding box coordinates
                const x = output[0 * stride + i];
                const y = output[1 * stride + i];
                const w = output[2 * stride + i];
                const h = output[3 * stride + i];
                
                // Convert from center coordinates to top-left
                const left = (x - w / 2) * (canvas.width / inputSize);
                const top = (y - h / 2) * (canvas.height / inputSize);
                const width = (w) * (canvas.width / inputSize);
                const height = (h) * (canvas.height / inputSize);
                
                predictions.push({
                    class: YOLOV8_CLASSES[classId] || 'Unknown',
                    score: maxConfidence,
                    bbox: [left, top, width, height]
                });
            }
        }
        
        // Draw detections
        drawDetections(predictions, 'yolov8');
    } catch (error) {
        console.error("YOLOv8 detection failed:", error);
    }
}

function drawDetections(results, modelType) {
    // Clear previous drawings by redrawing the canvas
    ctx1.drawImage(video, 0, 0, c1.width, c1.height);
    
    for (let index = 0; index < results.length; index++) {
        const element = results[index];
        let label, x, y, width, height, confidence;
        
        if (modelType === 'yolov8') {
            label = element.class || 'Unknown';
            x = element.bbox[0];
            y = element.bbox[1];
            width = element.bbox[2];
            height = element.bbox[3];
            confidence = element.score || 0;
        } else if (modelType === 'yolo') {
            label = element.class || 'Unknown';
            x = element.bbox[0];
            y = element.bbox[1];
            width = element.bbox[2];
            height = element.bbox[3];
            confidence = element.score || 0;
        } else {
            label = element.label || 'Unknown';
            x = element.x;
            y = element.y;
            width = element.width;
            height = element.height;
            confidence = element.confidence || 0;
        }
        
        // Draw label and confidence
        ctx1.font = "15px Arial";
        ctx1.fillStyle = "red";
        ctx1.fillText(label + " - " + (confidence * 100).toFixed(2) + "%", x + 10, y + 15);
        
        // Draw bounding box
        ctx1.beginPath();
        ctx1.strokeStyle = "red";
        ctx1.lineWidth = 2;
        ctx1.rect(x, y, width, height);
        ctx1.stroke();
    }
}