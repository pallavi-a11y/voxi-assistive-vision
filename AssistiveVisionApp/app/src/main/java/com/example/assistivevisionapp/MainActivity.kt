@file:OptIn(androidx.camera.core.ExperimentalGetImage::class)

package com.example.assistivevisionapp

import android.Manifest
import android.content.Context
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import android.graphics.ImageFormat
import android.graphics.Rect
import android.graphics.YuvImage
import android.os.Bundle
import android.os.Handler
import android.speech.RecognitionListener
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer
import android.speech.tts.TextToSpeech
import android.util.Log
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.LocalLifecycleOwner
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.text.TextRecognition
import com.google.mlkit.vision.text.latin.TextRecognizerOptions
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import java.io.ByteArrayOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.*
import kotlin.math.abs

class MainActivity : ComponentActivity() {
    private var textToSpeech: TextToSpeech? = null
    private var speechRecognizer: SpeechRecognizer? = null
    private lateinit var voiceIntent: Intent
    private var imageCapture: ImageCapture? = null
    private var selectedLang = "none"
    private var isListening = false
    private var isMicEnabled = true

    // TensorFlow Lite variables
    private var tflite: Interpreter? = null
    private var labels: List<String> = emptyList()
    private var isModelLoaded = false

    // Model input/output specifications
    private var inputSize = 300
    private var isQuantized = false

    // Data classes for smart detection
    data class ImageAnalysis(
        val hasScreenLikeArea: Boolean,
        val isLikelyIndoor: Boolean,
        val dominantColors: List<String>,
        val aspectRatio: Float,
        val brightnessLevel: String,
        val hasKeyboardPattern: Boolean
    )

    data class RawDetection(
        val classIndex: Int,
        val confidence: Float,
        val label: String
    )

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Initialize detector early
        initDetector(applicationContext)

        setContent {
            val context = LocalContext.current
            val lifecycleOwner = LocalLifecycleOwner.current
            var permissionsGranted by remember { mutableStateOf(false) }

            val permissionLauncher = rememberLauncherForActivityResult(
                ActivityResultContracts.RequestMultiplePermissions()
            ) { perms ->
                permissionsGranted = perms[Manifest.permission.RECORD_AUDIO] == true &&
                        perms[Manifest.permission.CAMERA] == true

                if (permissionsGranted) {
                    initTextToSpeech(context)
                    initSpeechRecognizer(context)
                }
            }

            LaunchedEffect(Unit) {
                permissionLauncher.launch(
                    arrayOf(Manifest.permission.RECORD_AUDIO, Manifest.permission.CAMERA)
                )
            }

            if (permissionsGranted) {
                CameraPreviewView(lifecycleOwner) { capture -> imageCapture = capture }
            } else {
                Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                    Text("Requesting permissions…", modifier = Modifier.padding(16.dp))
                }
            }
        }
    }

    private fun initDetector(context: Context) {
        try {
            Log.d("VoxiInit", "Starting model initialization...")

            // Load model file
            val model = FileUtil.loadMappedFile(context, "detect.tflite")
            val options = Interpreter.Options().apply {
                setNumThreads(4)
            }
            tflite = Interpreter(model, options)

            // Get input tensor shape
            val inputShape = tflite?.getInputTensor(0)?.shape()
            if (inputShape != null && inputShape.size >= 3) {
                inputSize = inputShape[1]
                Log.d("VoxiInit", "Input size detected: $inputSize")
            }

            // Check if model is quantized
            val inputTensor = tflite?.getInputTensor(0)
            isQuantized = inputTensor?.dataType() == org.tensorflow.lite.DataType.UINT8
            Log.d("VoxiInit", "Model is quantized: $isQuantized")

            // Load labels
            val labelStream = context.assets.open("labels.txt")
            labels = labelStream.bufferedReader().readLines()
            labelStream.close()

            isModelLoaded = true
            Log.d("VoxiInit", "Model loaded ✅, Labels loaded: ${labels.size}")

        } catch (e: Exception) {
            Log.e("VoxiInit", "Detector init failed: ${e.message}", e)
            isModelLoaded = false
        }
    }

    private fun initTextToSpeech(context: Context) {
        textToSpeech = TextToSpeech(context) {
            if (it == TextToSpeech.SUCCESS) {
                textToSpeech?.language = Locale("hi")
                val greeting = "hey there . i am voxi . your personal assistant . please select your language - english or hindi ." +
                        "मै Voxi हूँ। कृपया अपनी भाषा चुनें — हिंदी या English।"
                textToSpeech?.speak(greeting, TextToSpeech.QUEUE_FLUSH, null, "greeting")
            }
        }
    }

    private fun initSpeechRecognizer(context: Context) {
        speechRecognizer = SpeechRecognizer.createSpeechRecognizer(context)
        voiceIntent = Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH).apply {
            putExtra(
                RecognizerIntent.EXTRA_LANGUAGE_MODEL,
                RecognizerIntent.LANGUAGE_MODEL_FREE_FORM
            )
            putExtra(RecognizerIntent.EXTRA_LANGUAGE, Locale.getDefault())
            putExtra(RecognizerIntent.EXTRA_PARTIAL_RESULTS, true)
        }

        speechRecognizer?.setRecognitionListener(object : RecognitionListener {
            override fun onReadyForSpeech(params: Bundle?) {
                isListening = true
            }

            override fun onBeginningOfSpeech() {}
            override fun onEndOfSpeech() {
                isListening = false
            }

            override fun onError(error: Int) {
                isListening = false
                Log.e("VoxiRecognizer", "Error code: $error")
                restartListeningWithDelay()
            }

            override fun onRmsChanged(rmsdB: Float) {}
            override fun onBufferReceived(buffer: ByteArray?) {}
            override fun onPartialResults(partialResults: Bundle?) {}
            override fun onEvent(eventType: Int, params: Bundle?) {}

            override fun onResults(results: Bundle?) {
                isListening = false
                val command = results?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)
                    ?.joinToString(" ")?.lowercase(Locale.getDefault()) ?: ""

                Log.d("VoxiRecognizer", "Heard: $command")

                when (matchUserIntent(command)) {
                    "lang_hindi" -> {
                        selectedLang = "hi"
                        textToSpeech?.language = Locale("hi")
                        textToSpeech?.speak(
                            "आपने हिंदी चुनी है। Voxi तैयार है।",
                            TextToSpeech.QUEUE_FLUSH,
                            null,
                            "lang_hi"
                        )
                    }

                    "lang_english" -> {
                        selectedLang = "en"
                        textToSpeech?.language = Locale.US
                        textToSpeech?.speak(
                            "You selected English. Voxi is ready to assist.",
                            TextToSpeech.QUEUE_FLUSH,
                            null,
                            "lang_en"
                        )
                    }

                    "read_document" -> {
                        val msg =
                            if (selectedLang == "hi") "पढ़ना शुरू कर रही हूँ…" else "Reading the document now..."
                        textToSpeech?.speak(msg, TextToSpeech.QUEUE_FLUSH, null, "read_start")
                        captureAndReadText(applicationContext)
                    }

                    "detect_object" -> {
                        val msg =
                            if (selectedLang == "hi") "मैं देख रही हूँ..." else "Scanning the scene..."
                        textToSpeech?.speak(msg, TextToSpeech.QUEUE_FLUSH, null, "detect_start")
                        captureAndDetectObject(applicationContext)
                    }

                    "help" -> {
                        val msg = if (selectedLang == "hi")
                            "आप कह सकते हैं: 'हिंदी', 'English', 'पढ़ो दस्तावेज़', या 'यह क्या है'।"
                        else
                            "You can say: 'Hindi', 'English', 'read document', or 'what is this'."
                        textToSpeech?.speak(msg, TextToSpeech.QUEUE_FLUSH, null, "help")
                    }

                    "clarify" -> {
                        val msg =
                            if (selectedLang == "hi") "मैं फिर से कहती हूँ…" else "Let me repeat that…"
                        textToSpeech?.speak(msg, TextToSpeech.QUEUE_FLUSH, null, "repeat")
                    }

                    "mic_off" -> {
                        isMicEnabled = false
                        val msg = if (selectedLang == "hi")
                            "ठीक है, मैं अब नहीं सुन रही हूँ।"
                        else
                            "Alright, I'll stop listening for now."
                        textToSpeech?.speak(msg, TextToSpeech.QUEUE_FLUSH, null, "mic_off")
                    }

                    "mic_on" -> {
                        isMicEnabled = true
                        val msg = if (selectedLang == "hi")
                            "मैं फिर से सुन रही हूँ।"
                        else
                            "Listening resumed. I'm here."
                        textToSpeech?.speak(msg, TextToSpeech.QUEUE_FLUSH, null, "mic_on")
                        restartListeningWithDelay()
                    }

                    else -> {
                        val msg = if (selectedLang == "hi")
                            "माफ़ कीजिए, मैं पूरी तरह समझ नहीं पाई। आप फिर से बोल सकते हैं या 'मदद' कहकर विकल्प जान सकते हैं।"
                        else
                            "I'm sorry, I didn't quite understand. You can try again or say 'help' to hear your options."
                        textToSpeech?.speak(msg, TextToSpeech.QUEUE_FLUSH, null, "unknown")
                    }
                }

                restartListeningWithDelay()
            }
        })

        restartListeningWithDelay()
    }

    private fun restartListeningWithDelay() {
        if (!isListening && isMicEnabled) {
            Handler(mainLooper).postDelayed({
                speechRecognizer?.startListening(voiceIntent)
                isListening = true
            }, 1000)
        }
    }

    @Composable
    fun CameraPreviewView(
        lifecycleOwner: androidx.lifecycle.LifecycleOwner,
        onCaptureReady: (ImageCapture) -> Unit
    ) {
        val context = LocalContext.current
        val previewView = remember { PreviewView(context) }

        LaunchedEffect(Unit) {
            val cameraProvider = ProcessCameraProvider.getInstance(context).get()
            val preview = Preview.Builder().build().apply {
                setSurfaceProvider(previewView.surfaceProvider)
            }

            val capture = ImageCapture.Builder()
                .setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
                .build()

            cameraProvider.unbindAll()
            cameraProvider.bindToLifecycle(
                lifecycleOwner,
                CameraSelector.DEFAULT_BACK_CAMERA,
                preview,
                capture
            )

            onCaptureReady(capture)
        }

        AndroidView(factory = { previewView }, modifier = Modifier.fillMaxSize())
    }

    private fun captureAndReadText(context: Context) {
        val capture = imageCapture ?: return
        capture.takePicture(
            ContextCompat.getMainExecutor(context),
            object : ImageCapture.OnImageCapturedCallback() {
                @androidx.annotation.OptIn(ExperimentalGetImage::class)
                override fun onCaptureSuccess(imageProxy: ImageProxy) {
                    val mediaImage = imageProxy.image
                    val rotation = imageProxy.imageInfo.rotationDegrees
                    if (mediaImage != null) {
                        val inputImage = InputImage.fromMediaImage(mediaImage, rotation)
                        val recognizer =
                            TextRecognition.getClient(TextRecognizerOptions.DEFAULT_OPTIONS)
                        recognizer.process(inputImage)
                            .addOnSuccessListener {
                                val text = it.text
                                val spoken = if (text.isNotBlank()) {
                                    if (selectedLang == "hi") "मैंने पढ़ा: $text" else "Here's what I read: $text"
                                } else {
                                    if (selectedLang == "hi") "माफ़ कीजिए, कुछ पढ़ नहीं पाई।" else "Sorry, I couldn't read anything."
                                }
                                textToSpeech?.speak(
                                    spoken,
                                    TextToSpeech.QUEUE_FLUSH,
                                    null,
                                    "read_result"
                                )
                            }
                            .addOnFailureListener {
                                val failMsg =
                                    if (selectedLang == "hi") "पढ़ने में दिक्कत हुई।" else "Text reading failed."
                                textToSpeech?.speak(
                                    failMsg,
                                    TextToSpeech.QUEUE_FLUSH,
                                    null,
                                    "read_error"
                                )
                            }
                    }
                    imageProxy.close()
                }

                override fun onError(exception: ImageCaptureException) {
                    Toast.makeText(
                        context,
                        "Capture error: ${exception.message}",
                        Toast.LENGTH_SHORT
                    ).show()
                }
            }
        )
    }

    private fun captureAndDetectObject(context: Context) {
        val capture = imageCapture ?: return

        capture.takePicture(
            ContextCompat.getMainExecutor(context),
            object : ImageCapture.OnImageCapturedCallback() {
                override fun onCaptureSuccess(imageProxy: ImageProxy) {
                    val bitmap = imageProxy.toBitmapCompat()
                    if (bitmap == null) {
                        val msg = if (selectedLang == "hi") "छवि रूपांतरण विफल।" else "Image conversion failed."
                        textToSpeech?.speak(msg, TextToSpeech.QUEUE_FLUSH, null, "bitmap_error")
                        imageProxy.close()
                        return
                    }

                    val result = runSSDWithSmartPostProcessing(bitmap)
                    val message = if (selectedLang == "hi") "मैं देख रही हूँ: $result" else "I see: $result"
                    textToSpeech?.speak(message, TextToSpeech.QUEUE_FLUSH, null, "ssd_result")

                    imageProxy.close()
                }

                override fun onError(e: ImageCaptureException) {
                    Log.e("VoxiCapture", "Capture failed: ${e.message}", e)
                    val msg = if (selectedLang == "hi") "कैमरा त्रुटि हुई।" else "Camera error occurred."
                    textToSpeech?.speak(msg, TextToSpeech.QUEUE_FLUSH, null, "capture_error")
                }
            }
        )
    }

    private fun matchUserIntent(command: String): String {
        val cmd = command.trim().lowercase(Locale.getDefault())

        val micOffTriggers = listOf(
            "stop listening", "mic off", "pause listening", "quiet mode", "be quiet", "shut up"
        )
        val micOnTriggers = listOf(
            "start listening", "resume listening", "mic on", "listen again", "you can talk now"
        )
        val hindiTriggers = listOf("hindi", "हिंदी", "हिन्दी", "speak hindi", "switch to hindi")
        val englishTriggers = listOf("english", "अंग्रेज़ी", "speak english", "switch to english")
        val readTriggers = listOf(
            "read", "read this", "read the page", "scan this", "can you read",
            "tell me what's written", "पढ़ो", "दस्तावेज़ पढ़ो", "क्या लिखा है", "read aloud"
        )
        val objectTriggers = listOf(
            "what is this", "recognize this", "can you identify", "object detection",
            "whats in front of me", "what am i looking at", "यह क्या है", "देखो",
            "मेरे सामने क्या है", "figure this out", "identify object"
        )
        val helpTriggers = listOf(
            "help", "what can you do", "show commands", "need help", "how to use", "मदद", "सहायता"
        )
        val repeatTriggers = listOf(
            "repeat", "say again", "can you repeat", "i missed it", "फिर", "फिर से बोलो"
        )

        fun containsAny(keywords: List<String>) = keywords.any { it in cmd }

        return when {
            containsAny(micOffTriggers) -> "mic_off"
            containsAny(micOnTriggers) -> "mic_on"
            containsAny(hindiTriggers) -> "lang_hindi"
            containsAny(englishTriggers) -> "lang_english"
            containsAny(readTriggers) -> "read_document"
            containsAny(objectTriggers) -> "detect_object"
            containsAny(helpTriggers) -> "help"
            containsAny(repeatTriggers) -> "clarify"
            else -> "unknown"
        }
    }

    @androidx.annotation.OptIn(ExperimentalGetImage::class)
    fun ImageProxy.toBitmapCompat(): Bitmap? {
        return try {
            val image = this.image ?: return null

            when (image.format) {
                ImageFormat.YUV_420_888 -> {
                    val yPlane = image.planes[0]
                    val uPlane = image.planes[1]
                    val vPlane = image.planes[2]

                    val ySize = yPlane.buffer.remaining()
                    val uSize = uPlane.buffer.remaining()
                    val vSize = vPlane.buffer.remaining()

                    val nv21 = ByteArray(ySize + uSize + vSize)

                    yPlane.buffer.get(nv21, 0, ySize)
                    uPlane.buffer.get(nv21, ySize, uSize)
                    vPlane.buffer.get(nv21, ySize + uSize, vSize)

                    val yuvImage = YuvImage(nv21, ImageFormat.NV21, width, height, null)
                    val out = ByteArrayOutputStream()
                    yuvImage.compressToJpeg(Rect(0, 0, width, height), 85, out)
                    val imageBytes = out.toByteArray()

                    BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
                }
                ImageFormat.JPEG -> {
                    val buffer = image.planes[0].buffer
                    val bytes = ByteArray(buffer.remaining())
                    buffer.get(bytes)
                    BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
                }
                else -> {
                    Log.w("VoxiImage", "Unsupported image format: ${image.format}")
                    null
                }
            }
        } catch (e: Exception) {
            Log.e("VoxiImage", "Bitmap conversion failed: ${e.message}", e)
            null
        }
    }

    // SMART OBJECT DETECTION WITH POST-PROCESSING
// BULLETPROOF VERSION: Replace your runSSDWithSmartPostProcessing function with this simple, working version

// OPTIMIZED VERSION: Replace your runSSDWithSmartPostProcessing function with this faster version

    private fun runSSDWithSmartPostProcessing(bitmap: Bitmap): String {
        Log.d("VoxiSSD", "Starting optimized detection...")

        return try {
            // Quick checks
            if (!isModelLoaded || tflite == null) {
                return if (selectedLang == "hi") "मॉडल लोड नहीं हुआ।" else "Model not loaded."
            }

            if (labels.isEmpty()) {
                return if (selectedLang == "hi") "लेबल लोड नहीं हुए।" else "Labels not loaded."
            }

            Log.d("VoxiSSD", "Model and labels OK, processing image...")

            // Fast image processing
            val processedBitmap = Bitmap.createScaledBitmap(bitmap, inputSize, inputSize, true)
            Log.d("VoxiSSD", "Image scaled to ${inputSize}x${inputSize}")

            // Prepare input
            val inputBuffer = prepareOptimizedInput(processedBitmap)
            Log.d("VoxiSSD", "Input buffer prepared")

            // Prepare outputs
            val outputArrays = prepareSimpleOutputs()
            Log.d("VoxiSSD", "Output arrays prepared")

            // Run inference
            Log.d("VoxiSSD", "Running inference...")
            tflite?.runForMultipleInputsOutputs(arrayOf(inputBuffer), outputArrays)
            Log.d("VoxiSSD", "Inference completed")

            // FAST processing - no complex analysis
            val result = processFastResults(outputArrays, bitmap)
            Log.d("VoxiSSD", "Final result: $result")

            result

        } catch (e: Exception) {
            Log.e("VoxiSSD", "Detection failed: ${e.message}", e)
            if (selectedLang == "hi") "स्कैन में समस्या हुई।" else "Scanning failed."
        }
    }

    private fun prepareOptimizedInput(bitmap: Bitmap): ByteBuffer {
        Log.d("VoxiSSD", "Preparing optimized input...")

        val inputBuffer = if (isQuantized) {
            ByteBuffer.allocateDirect(inputSize * inputSize * 3)
        } else {
            ByteBuffer.allocateDirect(4 * inputSize * inputSize * 3)
        }

        inputBuffer.order(ByteOrder.nativeOrder())

        val pixels = IntArray(inputSize * inputSize)
        bitmap.getPixels(pixels, 0, inputSize, 0, 0, inputSize, inputSize)

        if (isQuantized) {
            for (pixel in pixels) {
                inputBuffer.put(Color.red(pixel).toByte())
                inputBuffer.put(Color.green(pixel).toByte())
                inputBuffer.put(Color.blue(pixel).toByte())
            }
        } else {
            for (pixel in pixels) {
                inputBuffer.putFloat(Color.red(pixel) / 255.0f)
                inputBuffer.putFloat(Color.green(pixel) / 255.0f)
                inputBuffer.putFloat(Color.blue(pixel) / 255.0f)
            }
        }

        Log.d("VoxiSSD", "Input ready, size: ${inputBuffer.capacity()}")
        return inputBuffer
    }

    private fun processFastResults(outputArrays: Map<Int, Any>, originalBitmap: Bitmap): String {
        Log.d("VoxiSSD", "Processing results quickly...")

        return try {
            val scores = outputArrays[2] as? Array<FloatArray> ?: return "Result error."
            val classes = outputArrays[1] as? Array<FloatArray> ?: return "Result error."
            val numDetectionsArray = outputArrays[3] as? FloatArray ?: return "Result error."

            val numDetections = numDetectionsArray[0].toInt().coerceIn(0, 10)
            Log.d("VoxiSSD", "Number of detections: $numDetections")

            val detections = mutableListOf<Pair<String, Int>>()

            // Process each detection quickly
            for (i in 0 until numDetections) {
                val score = scores[0][i]
                val classIndex = classes[0][i].toInt()

                Log.d("VoxiSSD", "Detection $i: score=$score, classIndex=$classIndex")

                if (score > 0.2f && classIndex >= 0 && classIndex < labels.size) {
                    var label = labels[classIndex].trim()
                    var finalScore = score

                    // SIMPLE refrigerator -> laptop conversion
                    if (label.lowercase().contains("refrigerator")) {
                        // Quick brightness check
                        if (hasSimpleLaptopIndicators(originalBitmap)) {
                            label = "laptop"
                            finalScore = score * 0.8f
                            Log.d("VoxiSSD", "✓ Converted refrigerator -> laptop")
                        }
                    }

                    // Simple label normalization
                    val displayLabel = when (label.lowercase().trim()) {
                        "cell phone", "mobile phone" -> "phone"
                        "television", "tv" -> "TV"
                        "dining table" -> "table"
                        else -> label
                    }

                    detections.add(Pair(displayLabel, (finalScore * 100).toInt()))
                    Log.d("VoxiSSD", "✓ Added: $displayLabel (${(finalScore * 100).toInt()}%)")
                }
            }

            // Quick result formatting
            if (detections.isEmpty()) {
                Log.d("VoxiSSD", "No valid detections")
                return if (selectedLang == "hi") "कोई वस्तु नहीं मिली।" else "No objects detected."
            }

            // Sort and take top 3
            val result = detections
                .sortedByDescending { it.second }
                .take(3)
                .joinToString(", ") { "${it.first} (${it.second}%)" }

            Log.d("VoxiSSD", "✓ Quick result: $result")
            return result

        } catch (e: Exception) {
            Log.e("VoxiSSD", "Fast processing error: ${e.message}")
            return if (selectedLang == "hi") "परिणाम त्रुटि।" else "Result error."
        }
    }

    private fun hasSimpleLaptopIndicators(bitmap: Bitmap): Boolean {
        return try {
            // VERY simple analysis - just check if there are bright areas
            val sampleSize = 100 // Only check 100 pixels for speed
            val step = (bitmap.width * bitmap.height) / sampleSize

            var brightPixels = 0
            var totalBrightness = 0

            for (i in 0 until sampleSize) {
                val x = (i * step) % bitmap.width
                val y = (i * step) / bitmap.width

                if (x < bitmap.width && y < bitmap.height) {
                    val pixel = bitmap.getPixel(x, y)
                    val brightness = (Color.red(pixel) + Color.green(pixel) + Color.blue(pixel)) / 3
                    totalBrightness += brightness

                    if (brightness > 150) brightPixels++
                }
            }

            val avgBrightness = totalBrightness / sampleSize
            val brightRatio = brightPixels.toFloat() / sampleSize

            // Simple heuristic: reasonable brightness + some bright areas = could be laptop
            val result = avgBrightness > 70 && brightRatio > 0.15f && brightRatio < 0.8f

            Log.d("VoxiSSD", "Simple laptop check: avgBright=$avgBrightness, brightRatio=$brightRatio, result=$result")

            result

        } catch (e: Exception) {
            Log.w("VoxiSSD", "Simple check failed: ${e.message}")
            false
        }
    }

    // Keep your existing prepareSimpleOutputs function - it's working fine 
    private fun prepareSimpleInput(bitmap: Bitmap): ByteBuffer {
        Log.d("VoxiSSD", "Preparing input buffer...")

        val inputBuffer = if (isQuantized) {
            ByteBuffer.allocateDirect(inputSize * inputSize * 3)
        } else {
            ByteBuffer.allocateDirect(4 * inputSize * inputSize * 3)
        }

        inputBuffer.order(ByteOrder.nativeOrder())

        val pixels = IntArray(inputSize * inputSize)
        bitmap.getPixels(pixels, 0, inputSize, 0, 0, inputSize, inputSize)

        if (isQuantized) {
            // Quantized input (0-255)
            for (pixel in pixels) {
                inputBuffer.put(Color.red(pixel).toByte())
                inputBuffer.put(Color.green(pixel).toByte())
                inputBuffer.put(Color.blue(pixel).toByte())
            }
        } else {
            // Float input (0-1)
            for (pixel in pixels) {
                inputBuffer.putFloat(Color.red(pixel) / 255.0f)
                inputBuffer.putFloat(Color.green(pixel) / 255.0f)
                inputBuffer.putFloat(Color.blue(pixel) / 255.0f)
            }
        }

        Log.d("VoxiSSD", "Input buffer ready, size: ${inputBuffer.capacity()}")
        return inputBuffer
    }

    private fun prepareSimpleOutputs(): Map<Int, Any> {
        Log.d("VoxiSSD", "Preparing output arrays...")

        // Standard COCO SSD outputs
        return mapOf(
            0 to Array(1) { Array(10) { FloatArray(4) } },  // locations [1, 10, 4]
            1 to Array(1) { FloatArray(10) },               // classes [1, 10]
            2 to Array(1) { FloatArray(10) },               // scores [1, 10]
            3 to FloatArray(1)                              // num_detections [1]
        )
    }

    private fun processSimpleResults(outputArrays: Map<Int, Any>, originalBitmap: Bitmap): String {
        Log.d("VoxiSSD", "Processing results...")

        return try {
            // Extract arrays safely
            val scores = outputArrays[2] as? Array<FloatArray> ?: run {
                Log.e("VoxiSSD", "Invalid scores array")
                return if (selectedLang == "hi") "परिणाम त्रुटि।" else "Result error."
            }

            val classes = outputArrays[1] as? Array<FloatArray> ?: run {
                Log.e("VoxiSSD", "Invalid classes array")
                return if (selectedLang == "hi") "परिणाम त्रुटि।" else "Result error."
            }

            val numDetectionsArray = outputArrays[3] as? FloatArray ?: run {
                Log.e("VoxiSSD", "Invalid num_detections array")
                return if (selectedLang == "hi") "परिणाम त्रुटि।" else "Result error."
            }

            val numDetections = numDetectionsArray[0].toInt().coerceIn(0, 10)
            Log.d("VoxiSSD", "Number of detections: $numDetections")

            val detections = mutableListOf<Triple<String, Float, Int>>()

            // Process each detection
            for (i in 0 until numDetections) {
                try {
                    val score = scores[0][i]
                    val classIndex = classes[0][i].toInt()

                    Log.d("VoxiSSD", "Detection $i: score=$score, classIndex=$classIndex")

                    if (score > 0.15f && classIndex >= 0 && classIndex < labels.size) {
                        var label = labels[classIndex].trim()
                        var finalScore = score

                        // Simple refrigerator -> laptop fix
                        if (label.lowercase().contains("refrigerator")) {
                            if (hasLaptopCharacteristics(originalBitmap)) {
                                label = "laptop"
                                finalScore = score * 0.8f
                                Log.d("VoxiSSD", "Converted refrigerator -> laptop")
                            }
                        }

                        val displayLabel = normalizeSimpleLabel(label)
                        detections.add(Triple(displayLabel, finalScore, (finalScore * 100).toInt()))
                        Log.d("VoxiSSD", "Added detection: $displayLabel (${(finalScore * 100).toInt()}%)")
                    }
                } catch (e: Exception) {
                    Log.w("VoxiSSD", "Error processing detection $i: ${e.message}")
                    continue
                }
            }

            // Format final result
            if (detections.isEmpty()) {
                Log.d("VoxiSSD", "No valid detections found")
                return if (selectedLang == "hi") "कोई वस्तु नहीं मिली।" else "No objects detected."
            }

            // Sort by confidence and take top 3
            val topDetections = detections
                .sortedByDescending { it.second }
                .take(3)

            val result = topDetections.joinToString(", ") { "${it.first} (${it.third}%)" }
            Log.d("VoxiSSD", "Final result: $result")

            return result

        } catch (e: Exception) {
            Log.e("VoxiSSD", "Error in processSimpleResults: ${e.message}", e)
            return if (selectedLang == "hi") "परिणाम प्रसंस्करण विफल।" else "Result processing failed."
        }
    }

    private fun hasLaptopCharacteristics(bitmap: Bitmap): Boolean {
        return try {
            // Simple brightness analysis for screen detection
            val pixels = IntArray(bitmap.width * bitmap.height)
            bitmap.getPixels(pixels, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)

            val avgBrightness = pixels.map { pixel ->
                (Color.red(pixel) + Color.green(pixel) + Color.blue(pixel)) / 3
            }.average()

            // Count bright pixels (potential screen area)
            val brightPixels = pixels.count { pixel ->
                val brightness = (Color.red(pixel) + Color.green(pixel) + Color.blue(pixel)) / 3
                brightness > 150
            }

            val brightRatio = brightPixels.toFloat() / pixels.size

            // Simple heuristic: if image has reasonable brightness and some bright areas, could be laptop
            val hasScreen = avgBrightness > 80 && brightRatio > 0.1f && brightRatio < 0.8f

            Log.d("VoxiSSD", "Laptop check - avgBrightness: $avgBrightness, brightRatio: $brightRatio, hasScreen: $hasScreen")

            hasScreen

        } catch (e: Exception) {
            Log.w("VoxiSSD", "Error in laptop characteristics check: ${e.message}")
            false
        }
    }

    private fun normalizeSimpleLabel(label: String): String {
        return when (label.lowercase().trim()) {
            "cell phone", "mobile phone", "smartphone" -> "phone"
            "television", "tv" -> "TV"
            "dining table" -> "table"
            "wine glass" -> "glass"
            "sports ball" -> "ball"
            "potted plant" -> "plant"
            "teddy bear" -> "teddy bear"
            "hair drier" -> "hair dryer"
            else -> label
        }
    }

    // DEBUGGING FUNCTION - Add this to help troubleshoot
    private fun debugModelInfo() {
        try {
            Log.d("VoxiDebug", "=== DEBUG INFO ===")
            Log.d("VoxiDebug", "Model loaded: $isModelLoaded")
            Log.d("VoxiDebug", "TFLite interpreter: ${tflite != null}")
            Log.d("VoxiDebug", "Labels count: ${labels.size}")
            Log.d("VoxiDebug", "Input size: $inputSize")
            Log.d("VoxiDebug", "Is quantized: $isQuantized")

            if (labels.size > 0) {
                Log.d("VoxiDebug", "First 10 labels: ${labels.take(10)}")
            }

            tflite?.let { interpreter ->
                val inputTensor = interpreter.getInputTensor(0)
                Log.d("VoxiDebug", "Input tensor shape: ${inputTensor.shape().contentToString()}")
                Log.d("VoxiDebug", "Input tensor type: ${inputTensor.dataType()}")

                for (i in 0 until interpreter.outputTensorCount) {
                    val outputTensor = interpreter.getOutputTensor(i)
                    Log.d("VoxiDebug", "Output $i shape: ${outputTensor.shape().contentToString()}")
                }
            }

            Log.d("VoxiDebug", "==================")
        } catch (e: Exception) {
            Log.e("VoxiDebug", "Debug failed: ${e.message}")
        }
    }

    // Call this in your initDetector function after loading the model:
// debugModelInfo()
    private fun analyzeImageForSmartDetection(bitmap: Bitmap): ImageAnalysis {
        val pixels = IntArray(bitmap.width * bitmap.height)
        bitmap.getPixels(pixels, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)

        // Calculate brightness
        val avgBrightness = pixels.map { pixel ->
            (Color.red(pixel) + Color.green(pixel) + Color.blue(pixel)) / 3
        }.average()

        val brightnessLevel = when {
            avgBrightness < 80 -> "dark"
            avgBrightness > 180 -> "bright"
            else -> "normal"
        }

        // Detect screen-like areas
        val hasScreenLikeArea = detectScreenPattern(pixels, bitmap.width, bitmap.height)

        // Detect keyboard pattern
        val hasKeyboardPattern = detectKeyboardPattern(pixels, bitmap.width, bitmap.height)

        // Indoor detection
        val isLikelyIndoor = avgBrightness < 160 && !hasHighBlueSky(pixels)

        // Dominant colors
        val dominantColors = analyzeDominantColors(pixels)

        val aspectRatio = bitmap.width.toFloat() / bitmap.height.toFloat()

        return ImageAnalysis(
            hasScreenLikeArea = hasScreenLikeArea,
            isLikelyIndoor = isLikelyIndoor,
            dominantColors = dominantColors,
            aspectRatio = aspectRatio,
            brightnessLevel = brightnessLevel,
            hasKeyboardPattern = hasKeyboardPattern
        )
    }

    private fun detectScreenPattern(pixels: IntArray, width: Int, height: Int): Boolean {
        var brightPixelCount = 0
        var consecutiveBrightLines = 0

        for (y in height/4 until 3*height/4 step 5) {
            var brightPixelsInLine = 0
            for (x in width/4 until 3*width/4) {
                val pixel = pixels[y * width + x]
                val brightness = (Color.red(pixel) + Color.green(pixel) + Color.blue(pixel)) / 3
                if (brightness > 150) {
                    brightPixelsInLine++
                }
            }

            if (brightPixelsInLine > width/8) {
                consecutiveBrightLines++
            }

            brightPixelCount += brightPixelsInLine
        }

        val brightRatio = brightPixelCount.toFloat() / (pixels.size / 4)
        return brightRatio > 0.15f && consecutiveBrightLines > 3
    }

    private fun detectKeyboardPattern(pixels: IntArray, width: Int, height: Int): Boolean {
        var horizontalPatterns = 0

        for (y in 2*height/3 until height-10 step 8) {
            var darkToLightTransitions = 0
            var lastBrightness = 0

            for (x in width/4 until 3*width/4 step 4) {
                val pixel = pixels[y * width + x]
                val brightness = (Color.red(pixel) + Color.green(pixel) + Color.blue(pixel)) / 3

                if (abs(brightness - lastBrightness) > 30) {
                    darkToLightTransitions++
                }
                lastBrightness = brightness
            }

            if (darkToLightTransitions > 5) {
                horizontalPatterns++
            }
        }

        return horizontalPatterns > 2
    }

    private fun hasHighBlueSky(pixels: IntArray): Boolean {
        val bluePixelCount = pixels.count { pixel ->
            val blue = Color.blue(pixel)
            val red = Color.red(pixel)
            val green = Color.green(pixel)
            blue > red + 20 && blue > green + 20 && blue > 120
        }
        return bluePixelCount.toFloat() / pixels.size > 0.25f
    }

    private fun analyzeDominantColors(pixels: IntArray): List<String> {
        val colorCounts = mutableMapOf<String, Int>()

        pixels.forEach { pixel ->
            val red = Color.red(pixel)
            val green = Color.green(pixel)
            val blue = Color.blue(pixel)

            val dominantColor = when {
                red > green + 30 && red > blue + 30 -> "red"
                green > red + 30 && green > blue + 30 -> "green"
                blue > red + 30 && blue > green + 30 -> "blue"
                red > 200 && green > 200 && blue > 200 -> "white"
                red < 80 && green < 80 && blue < 80 -> "black"
                abs(red - green) < 30 && abs(green - blue) < 30 -> "gray"
                else -> "mixed"
            }

            colorCounts[dominantColor] = colorCounts.getOrDefault(dominantColor, 0) + 1
        }

        return colorCounts.entries
            .sortedByDescending { it.value }
            .take(3)
            .map { it.key }
    }

    private fun runStandardDetection(bitmap: Bitmap): List<RawDetection> {
        val processedBitmap = Bitmap.createScaledBitmap(bitmap, inputSize, inputSize, true)

        val inputBuffer = if (isQuantized) {
            prepareQuantizedInput(processedBitmap)
        } else {
            prepareFloatInput(processedBitmap)
        }

        val outputArrays = prepareOutputArrays()
        tflite?.runForMultipleInputsOutputs(arrayOf(inputBuffer), outputArrays)

        return extractRawDetections(outputArrays)
    }

    private fun prepareFloatInput(bitmap: Bitmap): ByteBuffer {
        val inputBuffer = ByteBuffer.allocateDirect(4 * inputSize * inputSize * 3)
        inputBuffer.order(ByteOrder.nativeOrder())

        val pixels = IntArray(inputSize * inputSize)
        bitmap.getPixels(pixels, 0, inputSize, 0, 0, inputSize, inputSize)

        for (pixel in pixels) {
            // Normalize to [-1, 1] range
            val red = (Color.red(pixel) - 127.5f) / 127.5f
            val green = (Color.green(pixel) - 127.5f) / 127.5f
            val blue = (Color.blue(pixel) - 127.5f) / 127.5f

            inputBuffer.putFloat(red)
            inputBuffer.putFloat(green)
            inputBuffer.putFloat(blue)
        }

        return inputBuffer
    }

    private fun prepareQuantizedInput(bitmap: Bitmap): ByteBuffer {
        val inputBuffer = ByteBuffer.allocateDirect(inputSize * inputSize * 3)
        inputBuffer.order(ByteOrder.nativeOrder())

        val pixels = IntArray(inputSize * inputSize)
        bitmap.getPixels(pixels, 0, inputSize, 0, 0, inputSize, inputSize)

        for (pixel in pixels) {
            inputBuffer.put(Color.red(pixel).toByte())
            inputBuffer.put(Color.green(pixel).toByte())
            inputBuffer.put(Color.blue(pixel).toByte())
        }

        return inputBuffer
    }

    private fun prepareOutputArrays(): Map<Int, Any> {
        return mapOf(
            0 to Array(1) { Array(10) { FloatArray(4) } },  // boxes
            1 to Array(1) { FloatArray(10) },               // classes
            2 to Array(1) { FloatArray(10) },               // scores
            3 to FloatArray(1)                              // num_detections
        )
    }

    private fun extractRawDetections(outputArrays: Map<Int, Any>): List<RawDetection> {
        val detections = mutableListOf<RawDetection>()

        try {
            @Suppress("UNCHECKED_CAST")
            val scores = outputArrays[2] as Array<FloatArray>
            @Suppress("UNCHECKED_CAST")
            val classes = outputArrays[1] as Array<FloatArray>
            val numDetections = (outputArrays[3] as FloatArray)[0].toInt().coerceAtMost(10)

            for (i in 0 until numDetections) {
                val score = scores[0][i]
                val classIndex = classes[0][i].toInt()

                if (score > 0.1f && classIndex >= 0 && classIndex < labels.size) {
                    val label = labels[classIndex]
                    detections.add(RawDetection(classIndex, score, label))
                    Log.d("VoxiSSD", "Raw detection: $label (${(score * 100).toInt()}%) - index: $classIndex")
                }
            }

        } catch (e: Exception) {
            Log.e("VoxiSSD", "Error extracting raw detections: ${e.message}")
        }

        return detections
    }

    private fun applySmartPostProcessing(
        rawDetections: List<RawDetection>,
        imageAnalysis: ImageAnalysis
    ): String {

        Log.d("VoxiSSD", "=== SMART POST-PROCESSING ===")
        Log.d("VoxiSSD", "Raw detections: ${rawDetections.map { "${it.label}(${(it.confidence * 100).toInt()}%)" }}")

        val smartDetections = mutableListOf<Pair<String, Float>>()

        // Process each detection with smart logic
        rawDetections.forEach { detection ->
            val smartResult = processDetectionWithContext(detection, imageAnalysis)
            if (smartResult != null) {
                smartDetections.add(smartResult)
            }
        }

        // If no smart detections, fall back to highest confidence raw detection
        if (smartDetections.isEmpty() && rawDetections.isNotEmpty()) {
            val fallback = rawDetections.maxByOrNull { it.confidence }!!
            smartDetections.add(Pair(fallback.label, fallback.confidence))
            Log.d("VoxiSSD", "Using fallback: ${fallback.label}")
        }

        // Format final result
        return formatSmartResult(smartDetections)
    }

    private fun processDetectionWithContext(
        detection: RawDetection,
        imageAnalysis: ImageAnalysis
    ): Pair<String, Float>? {

        val originalLabel = detection.label.lowercase()
        var finalLabel = detection.label
        var finalConfidence = detection.confidence

        // SMART RULE 1: Refrigerator → Laptop when laptop characteristics present
        if (originalLabel.contains("refrigerator")) {
            val laptopScore = calculateLaptopLikelihood(imageAnalysis)
            Log.d("VoxiSSD", "Refrigerator detected, laptop likelihood: $laptopScore")

            if (laptopScore > 0.6f) {
                finalLabel = "laptop"
                finalConfidence = detection.confidence * 0.85f // Slight penalty for conversion
                Log.d("VoxiSSD", "✓ Converted refrigerator → laptop (score: $laptopScore)")
            }
        }

        // SMART RULE 2: Microwave → Laptop when screen detected
        if (originalLabel.contains("microwave") && imageAnalysis.hasScreenLikeArea) {
            finalLabel = "laptop"
            finalConfidence = detection.confidence * 0.8f
            Log.d("VoxiSSD", "✓ Converted microwave → laptop (screen detected)")
        }

        // SMART RULE 3: Boost confidence for contextually appropriate objects
        when (originalLabel) {
            "laptop", "computer", "notebook" -> {
                if (imageAnalysis.hasScreenLikeArea || imageAnalysis.hasKeyboardPattern) {
                    finalConfidence *= 1.3f
                    Log.d("VoxiSSD", "✓ Boosted laptop confidence due to screen/keyboard pattern")
                }
            }

            "cell phone", "mobile phone" -> {
                if (imageAnalysis.aspectRatio > 1.5f && imageAnalysis.hasScreenLikeArea) {
                    finalConfidence *= 1.2f
                    Log.d("VoxiSSD", "✓ Boosted phone confidence")
                }
            }

            "person" -> {
                finalConfidence *= 1.1f // People detection is usually reliable
            }

            "cup", "bottle" -> {
                if (imageAnalysis.isLikelyIndoor) {
                    finalConfidence *= 1.15f
                    Log.d("VoxiSSD", "✓ Boosted cup/bottle confidence for indoor scene")
                }
            }
        }

        // Only return if confidence is reasonable
        return if (finalConfidence > 0.15f) {
            Pair(finalLabel, minOf(finalConfidence, 1.0f))
        } else {
            null
        }
    }

    private fun calculateLaptopLikelihood(imageAnalysis: ImageAnalysis): Float {
        var score = 0.0f

        // Screen presence is strongest indicator
        if (imageAnalysis.hasScreenLikeArea) score += 0.4f

        // Keyboard pattern
        if (imageAnalysis.hasKeyboardPattern) score += 0.3f

        // Indoor setting
        if (imageAnalysis.isLikelyIndoor) score += 0.1f

        // Appropriate colors (gray, black, white common for laptops)
        if (imageAnalysis.dominantColors.any { it in listOf("gray", "black", "white") }) {
            score += 0.15f
        }

        // Brightness level (laptops usually have some screen brightness)
        if (imageAnalysis.brightnessLevel == "normal" || imageAnalysis.brightnessLevel == "bright") {
            score += 0.1f
        }

        return minOf(score, 1.0f)
    }

    private fun formatSmartResult(detections: List<Pair<String, Float>>): String {
        if (detections.isEmpty()) {
            return if (selectedLang == "hi") "कोई वस्तु नहीं मिली।" else "No objects detected."
        }

        // Group similar detections and take highest confidence
        val groupedDetections = detections
            .groupBy { normalizeLabel(it.first) }
            .mapValues { (_, group) -> group.maxByOrNull { it.second }!! }
            .values
            .filter { it.second > 0.2f }
            .sortedByDescending { it.second }
            .take(3)

        return if (groupedDetections.isNotEmpty()) {
            val result = groupedDetections.joinToString(", ") {
                "${normalizeLabel(it.first)} (${(it.second * 100).toInt()}%)"
            }
            Log.d("VoxiSSD", "✓ Smart final result: $result")
            result
        } else {
            if (selectedLang == "hi") "कोई स्पष्ट वस्तु नहीं दिखी।" else "No clear objects visible."
        }
    }

    private fun normalizeLabel(label: String): String {
        val enhancedLabelMap = mapOf(
            // Electronics
            "laptop" to "laptop", "notebook" to "laptop", "computer" to "laptop",
            "cell phone" to "phone", "mobile phone" to "phone", "smartphone" to "phone",
            "television" to "TV", "tv" to "TV", "monitor" to "screen", "display" to "screen",

            // Bags and accessories
            "handbag" to "bag", "backpack" to "bag", "suitcase" to "bag", "purse" to "bag",
            "briefcase" to "bag", "luggage" to "bag",

            // Furniture
            "chair" to "chair", "couch" to "sofa", "sofa" to "sofa",
            "dining table" to "table", "table" to "table",

            // Vehicles
            "car" to "car", "automobile" to "car", "vehicle" to "car",
            "bicycle" to "bike", "motorbike" to "bike", "motorcycle" to "bike",
            "bus" to "bus", "truck" to "truck",

            // Food and drink
            "bottle" to "bottle", "wine bottle" to "bottle", "water bottle" to "bottle",
            "cup" to "cup", "mug" to "cup", "glass" to "cup",
            "bowl" to "bowl", "plate" to "plate",

            // Common objects
            "book" to "book", "newspaper" to "book", "magazine" to "book",
            "clock" to "clock", "watch" to "watch",
            "umbrella" to "umbrella", "glasses" to "glasses",
            "remote" to "remote", "mouse" to "mouse", "keyboard" to "keyboard",

            // People and animals
            "person" to "person", "people" to "person", "man" to "person", "woman" to "person",
            "dog" to "dog", "cat" to "cat", "bird" to "bird"
        )

        val lowerLabel = label.lowercase().trim()
        return enhancedLabelMap[lowerLabel] ?: label
    }

    override fun onDestroy() {
        super.onDestroy()
        textToSpeech?.shutdown()
        speechRecognizer?.destroy()
        tflite?.close()
    }
}