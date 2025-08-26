# FACE-DETECTION

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Face Detection & Recognition</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script async src="https://docs.opencv.org/4.5.5/opencv.js" onload="onOpenCvReady();" type="text/javascript"></script>
    <style>
        #videoContainer {
            position: relative;
            max-width: 800px;
            margin: 0 auto;
        }
        #videoElement {
            background-color: #666;
            transform: scaleX(-1); /* Flip camera horizontally */
        }
        #canvasOutput {
            position: absolute;
            top: 0;
            left: 0;
            pointer-events: none;
        }
        #loading {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.8);
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            z-index: 1000;
            color: white;
        }
        .face-box {
            position: absolute;
            border: 3px solid #3B82F6;
            border-radius: 4px;
            background-color: rgba(59, 130, 246, 0.2);
        }
        .face-label {
            position: absolute;
            bottom: -25px;
            left: 0;
            color: white;
            background-color: rgba(0, 0, 0, 0.7);
            padding: 2px 5px;
            font-size: 12px;
            border-radius: 3px;
        }
    </style>
</head>
<body>
    <div id="loading">Loading OpenCV.js...</div>
    <div class="container mx-auto px-4 py-8">
        <header class="text-center mb-8">
            <h1 class="text-4xl font-bold text-green-500 mb-2">AI Face Detection & Recognition</h1>
            <p class="text-gray-300">Real-time face detection with Haar cascades and deep learning</p>
        </header>

        <div class="flex flex-col lg:flex-row gap-8">
            <div class="lg:w-2/3">
                <div class="bg-gray-900 rounded-lg overflow-hidden shadow-xl mb-4">
                    <div id="videoContainer" class="w-full">
                        <video id="videoElement" width="800" height="600" autoplay muted class="w-full rounded-lg"></video>
                        <canvas id="canvasOutput" width="800" height="600"></canvas>
                    </div>
                </div>

                <div class="bg-gray-900 rounded-lg p-6 shadow-xl">
                    <div class="flex flex-wrap gap-4 mb-4">
                        <button id="startCamera" class="bg-green-600 hover:bg-green-700 text-white px-4 py-2 rounded-lg transition">
                            <i class="fas fa-video"></i> Start Camera
                        </button>
                        <button id="stopCamera" disabled class="bg-red-600 hover:bg-red-700 text-white px-4 py-2 rounded-lg transition">
                            <i class="fas fa-stop"></i> Stop Camera
                        </button>
                        <button id="captureImage" disabled class="bg-yellow-600 hover:bg-yellow-700 text-white px-4 py-2 rounded-lg transition">
                            <i class="fas fa-camera"></i> Capture Image
                        </button>
                        <button id="detectFaces" disabled class="bg-purple-600 hover:bg-purple-700 text-white px-4 py-2 rounded-lg transition">
                            <i class="fas fa-search"></i> Detect Faces
                        </button>
                    </div>

                    <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <div>
                            <label class="block text-sm font-medium mb-2">Detection Model</label>
                            <select id="modelSelect" class="w-full bg-gray-800 border-gray-600 text-white rounded-lg px-3 py-2">
                                <option value="haar">Haar Cascade</option>
                                <option value="lbp">LBP Cascade</option>
                            </select>
                        </div>
                        <div>
                            <label class="block text-sm font-medium mb-2">Detection Scale</label>
                            <input id="scaleInput" type="range" min="1.05" max="1.5" step="0.05" value="1.2" class="w-full">
                            <span id="scaleValue" class="text-sm">1.2</span>
                        </div>
                    </div>
                </div>
            </div>

            <div class="lg:w-1/3">
                <div class="bg-gray-900 rounded-lg p-6 shadow-xl">
                    <h2 class="text-xl font-semibold mb-4 text-green-500">Detection Results</h2>
                    <div id="detectionResults" class="space-y-4">
                        <div class="p-4 bg-gray-800 rounded-lg">
                            <p class="text-gray-400">No faces detected yet</p>
                        </div>
                    </div>
                </div>

                <div class="bg-gray-900 rounded-lg p-6 shadow-xl mt-4">
                    <h2 class="text-xl font-semibold mb-4 text-green-500">Captured Faces</h2>
                    <div id="capturedFaces" class="grid grid-cols-2 gap-2">
                        <div class="p-2 bg-gray-800 rounded-lg text-center">
                            <img src="https://placehold.co/150x150" alt="Placeholder for detected faces" class="w-full rounded">
                            <p class="text-sm mt-1 text-gray-400">No face captured</p>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <div class="mt-8 bg-gray-900 rounded-lg p-6 shadow-xl">
            <h2 class="text-xl font-semibold mb-4 text-green-500">About This System</h2>
            <div class="text-gray-300 space-y-4">
                <p>This application demonstrates real-time face detection using computer vision techniques:</p>
                <ul class="list-disc pl-5 space-y-2">
                    <li>Uses OpenCV.js with pre-trained Haar and LBP cascade classifiers</li>
                    <li>Processes video feed from your webcam directly in the browser</li>
                    <li>Visualizes detected faces with bounding boxes</li>
                    <li>No server-side processing required - all computation happens locally</li>
                </ul>
            </div>
        </div>
    </div>

    <script>
        // Global variables
        let video = document.getElementById('videoElement');
        let canvas = document.getElementById('canvasOutput');
        let ctx = canvas.getContext('2d');
        let streaming = false;
        let faceDetectionInterval = null;
        let haarModel = null;
        let lbpModel = null;
        
        // UI elements
        const startButton = document.getElementById('startCamera');
        const stopButton = document.getElementById('stopCamera');
        const captureButton = document.getElementById('captureImage');
        const detectButton = document.getElementById('detectFaces');
        const modelSelect = document.getElementById('modelSelect');
        const scaleInput = document.getElementById('scaleInput');
        const scaleValue = document.getElementById('scaleValue');
        const detectionResults = document.getElementById('detectionResults');
        const capturedFaces = document.getElementById('capturedFaces');
        const loading = document.getElementById('loading');

        // Update scale value display
        scaleInput.addEventListener('input', () => {
            scaleValue.textContent = scaleInput.value;
        });

        // OpenCV.js ready callback
        function onOpenCvReady() {
            loading.style.display = 'none';
            console.log('OpenCV.js is ready');
            
            // Setup camera controls
            setupCameraControls();
        }

        function setupCameraControls() {
            // Start camera button click handler
            startButton.addEventListener('click', async () => {
                try {
                    const stream = await navigator.mediaDevices.getUser Media({
                        video: {
                            width: { ideal: 1280 },
                            height: { ideal: 720 }
                        },
                        audio: false
                    });
                    video.srcObject = stream;
                    video.play();
                    streaming = true;
                    
                    // Enable/disable buttons
                    startButton.disabled = true;
                    stopButton.disabled = false;
                    captureButton.disabled = false;
                    detectButton.disabled = false;
                    
                } catch (err) {
                    console.error('Error accessing camera:', err);
                    alert('Could not access the camera. Please ensure you have granted camera permissions.');
                }
            });

            // Stop camera button click handler
            stopButton.addEventListener('click', () => {
                if (video.srcObject) {
                    video.srcObject.getTracks().forEach(track => track.stop());
                    video.srcObject = null;
                    streaming = false;
                    
                    // Clear any detection interval
                    if (faceDetectionInterval) {
                        clearInterval(faceDetectionInterval);
                        faceDetectionInterval = null;
                    }
                    
                    // Clear canvas
                    ctx.clearRect(0, 0, canvas.width, canvas.height);
                    
                    // Enable/disable buttons
                    startButton.disabled = false;
                    stopButton.disabled = true;
                    captureButton.disabled = true;
                    detectButton.disabled = true;
                }
            });

            // Capture image button click handler
            captureButton.addEventListener('click', () => {
                if (streaming) {
                    // Get current video frame
                    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                    
                    // Create image from canvas
                    const imageDataUrl = canvas.toDataURL('image/png');
                    
                    // Add to captured faces
                    const faceBox = document.createElement('div');
                    faceBox.className = 'p-2 bg-gray-800 rounded-lg text-center';
                    faceBox.innerHTML = `
                        <img src="${imageDataUrl}" alt="Captured face" class="w-full rounded">
                        <p class="text-sm mt-1">Captured at ${new Date().toLocaleTimeString()}</p>
                    `;
                    capturedFaces.prepend(faceBox);
                }
            });

            // Detect faces button click handler
            detectButton.addEventListener('click', () => {
                if (!streaming) return;
                
                // Load models if not already loaded
                if (!haarModel) {
                    haarModel = new cv.CascadeClassifier();
                    haarModel.load('haarcascade_frontalface_default.xml');
                }
                
                if (!lbpModel) {
                    lbpModel = new cv.CascadeClassifier();
                    lbpModel.load('lbpcascade_frontalface_improved.xml');
                }
                
                // Toggle detection
                if (faceDetectionInterval) {
                    clearInterval(faceDetectionInterval);
                    faceDetectionInterval = null;
                    detectButton.textContent = 'Detect Faces';
                    ctx.clearRect(0, 0, canvas.width, canvas.height);
                } else {
                    detectButton.textContent = 'Stop Detection';
                    faceDetectionInterval = setInterval(detectFaces, 100);
                }
            });
        }

        // Main face detection function
        function detectFaces() {
            if (!streaming) return;
            
            try {
                // Create video frame and canvas
                ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                
                // Convert to OpenCV format
                const src = cv.imread('canvasOutput');
                const gray = new cv.Mat();
                cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);
                
                // Detect faces
                const faces = new cv.RectVector();
                const selectedModel = modelSelect.value === 'haar' ? haarModel : lbpModel;
                const scaleFactor = parseFloat(scaleInput.value);
                const minNeighbors = 3;
                const minSize = new cv.Size(30, 30);
                
                selectedModel.detectMultiScale(gray, faces, scaleFactor, minNeighbors, 0, minSize);
                
                // Clear previous results
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                
                // Draw detected faces
                const faceElements = [];
                
                for (let i = 0; i < faces.size(); ++i) {
                    const faceRect = faces.get(i);
                    
                    // Draw rectangle
                    ctx.strokeStyle = '#3B82F6';
                    ctx.lineWidth = 3;
                    ctx.strokeRect(faceRect.x, faceRect.y, faceRect.width, faceRect.height);
                    
                    // Add face to results list
                    faceElements.push(`
                        <div class="p-2 bg-gray-800 rounded-lg">
                            <div class="text-blue-400 font-medium">Face ${i+1}</div>
                            <div class="text-sm text-gray-300 mt-1">
                                Position: (${faceRect.x}, ${faceRect.y})<br>
                                Size: ${faceRect.width}x${faceRect.height} px
                            </div>
                        </div>
                    `);
                }
                
                // Update results display
                if (faceElements.length > 0) {
                    detectionResults.innerHTML = faceElements.join('');
                } else {
                    detectionResults.innerHTML = `
                        <div class="p-4 bg-gray-800 rounded-lg">
                            <p class="text-gray-400">No faces detected</p>
                        </div>
                    `;
                }
                
                // Clean up
                src.delete();
                gray.delete();
                faces.delete();
                
            } catch (err) {
                console.error('Error in face detection:', err);
            }
        }

        // Initialize OpenCV.js if it's already loaded (for page reloads)
        if (window.cv) {
            onOpenCvReady();
        }
    </script>
</body>
</html>
