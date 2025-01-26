const statusElement = document.getElementById("status");
const video = document.getElementById("webcam");
const canvas = document.getElementById("snapshot");
const inputElement = document.getElementById("input");

let recognizing = false;

const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
const recognition = new SpeechRecognition();
recognition.lang = "ar-SA";
recognition.interimResults = false;
recognition.continuous = true;

// Start recognition when the page loads
window.onload = () => {
  recognition.start();
  statusElement.textContent = "Listening for keyword 'يا عدسه'...";
};

// Access the webcam
navigator.mediaDevices.getUserMedia({ video: true }).then((stream) => {
  video.srcObject = stream;
  video.style.display = "block";
});

// Process speech recognition results
recognition.onresult = async (event) => {
  const transcriptObj = event.results[event.results.length - 1][0];
  const transcript = transcriptObj.transcript.trim();
  const confidence = transcriptObj.confidence;

  console.log("Transcript:", transcript, "Confidence:", confidence);
  inputElement.textContent = transcript;

  if (confidence > 0.7 && transcript.toLowerCase().includes("يا عدسه")) {
    statusElement.textContent = "Keyword 'ياعدسه' detected. Capturing image...";
    recognition.stop();

    const context = canvas.getContext("2d");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    context.drawImage(video, 0, 0, canvas.width, canvas.height);

    canvas.toBlob(async (blob) => {
      const formData = new FormData();
      formData.append("file", blob, "snapshot.jpg");
      formData.append("prompt", transcript);

      try {
        const response = await fetch("http://localhost:8000/process", {
          method: "POST",
          body: formData,
        });

        if (response.ok) {
          const data = await response.json();
          console.log("Response:", data);
          await speakArabicText(data.result); // Use ElevenLabs TTS
        } else {
          const errorMessage = await response.text();
          console.error("Error:", response.status, errorMessage);
          alert(`Error: ${response.status} - ${errorMessage}`);
        }
      } catch (error) {
        console.error("Error:", error);
        alert("An error occurred. Check the console for details.");
      }

      recognition.start(); // Restart recognition
    }, "image/jpeg");
  }
};

// ElevenLabs TTS Function
async function useElevenLabsTTS(text) {
  const apiKey = ""; // Replace with your ElevenLabs API key
  const voiceId = "EXAVITQu4vr4xnSDxMaL"; // Replace with your ElevenLabs voice ID

  try {
    const response = await fetch(`https://api.elevenlabs.io/v1/text-to-speech/${voiceId}`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "xi-api-key": apiKey,
      },
      body: JSON.stringify({
        text: text,
        model_id: "eleven_turbo_v2_5",
        voice_settings: {
          stability: 0.5,
          similarity_boost: 0.7,
        },
      }),
    });

    if (!response.ok) {
      throw new Error(`Error: ${response.status} - ${response.statusText}`);
    }

    const audioBlob = await response.blob();
    const audioUrl = URL.createObjectURL(audioBlob);
    const audio = new Audio(audioUrl);

    // Play audio and resume recognition after it finishes
    audio.onended = () => {
      console.log("Audio finished. Resuming speech recognition...");
      // recognition.start();
    };

    audio.play();
  } catch (error) {
    console.error("Error with ElevenLabs TTS:", error);
    recognition.start(); // Resume recognition even if TTS fails
  }
}

// Function to handle Arabic text-to-speech
async function speakArabicText(text) {
  await useElevenLabsTTS(text); // Use ElevenLabs API
}

// Example: Display and read Arabic results
function displayArabicResult(resultText) {
  const resultElement = document.getElementById("result");
  resultElement.textContent = resultText;

  // Read the result aloud
  speakArabicText(resultText);
}
