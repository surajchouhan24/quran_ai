let mediaRecorder;
let audioChunks = [];

const recordBtn = document.getElementById('recordBtn');
const statusText = document.getElementById('status');
const quranDisplay = document.getElementById('quranDisplay');

// STEP 1: Upload Image
async function uploadImage() {
    const fileInput = document.getElementById('imageInput');
    if (!fileInput.files[0]) return alert("Please select an image");

    const formData = new FormData();
    formData.append('file', fileInput.files[0]);

    statusText.textContent = "Processing OCR...";
    
    try {
        // Changed to relative path for better compatibility
        const response = await fetch('/upload-page', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) throw new Error('OCR failed');
        
        const data = await response.json();
        
        // Display the words as individual spans
        quranDisplay.innerHTML = data.reference.split(/\s+/)
            .map(word => `<span class="word">${word}</span>`).join(' ');
        
        recordBtn.disabled = false;
        statusText.textContent = "Ready! Click Start Reciting.";
    } catch (err) {
        console.error(err);
        statusText.textContent = "Error: Could not extract text from image.";
    }
}

// STEP 2: Record Audio
recordBtn.onclick = async () => {
    // Check if recording is active
    if (mediaRecorder && mediaRecorder.state === "recording") {
        mediaRecorder.stop();
        recordBtn.textContent = "Start Reciting";
        statusText.textContent = "Analyzing Audio...";
        recordBtn.classList.remove('recording');
    } else {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            mediaRecorder = new MediaRecorder(stream);
            audioChunks = [];

            mediaRecorder.ondataavailable = (e) => {
                if (e.data.size > 0) audioChunks.push(e.data);
            };
            
            mediaRecorder.onstop = sendAudioToServer;

            mediaRecorder.start();
            recordBtn.textContent = "Stop & Check";
            recordBtn.classList.add('recording');
            statusText.textContent = "Listening to your recitation...";
        } catch (err) {
            console.error(err);
            alert("Microphone access denied or not found.");
        }
    }
};

async function sendAudioToServer() {
    // Create a proper blob with a mimetype the server expects
    const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
    const formData = new FormData();
    formData.append('file', audioBlob, 'recitation.wav');

    try {
        // Changed to relative path
        const response = await fetch('/analyze-recitation', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) throw new Error('Analysis failed');

        const data = await response.json();
        displayResults(data);
    } catch (err) {
        console.error(err);
        statusText.textContent = "Error: Analysis failed. Please try again.";
    }
}

function displayResults(data) {
    const transContainer = document.getElementById('transcription');
    if (transContainer) transContainer.textContent = data.transcription;
    
    // Color code the words
    quranDisplay.innerHTML = data.feedback.map(item => {
        let color = "#2c3e50"; // Default dark blue
        if (item.status === "correct") color = "#27ae60"; // Green
        if (item.status === "incorrect") color = "#e74c3c"; // Red
        if (item.status === "missing") color = "#bdc3c7"; // Grey
        
        return `<span style="color: ${color}; margin: 0 5px; font-weight: bold;">${item.word}</span>`;
    }).join(' ');

    statusText.textContent = data.is_perfect ? "Excellent! No mistakes." : "Mistakes detected. Try again!";
}