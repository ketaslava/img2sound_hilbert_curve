#include <libinput.h>
#include <linux/input.h>
#include <fcntl.h>
#include <ostream>
#include <unistd.h>
#include <iostream>
#include <chrono>
#include <thread>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <array>
#include <string>
#include <sstream>
#include <algorithm>
#include <cstring>
#include <random>
#include <fstream>
#include <iostream>
#include <iostream>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <fstream>
#include <iterator>
#include <thread>
#include <chrono>
#include <portaudio.h>
#include <deque>
#include <mutex>
#include <condition_variable>
#include <alsa/asoundlib.h>


// Namespace


using namespace cv;
using namespace std;


// Settings


// Constants
constexpr int SAMPLE_RATE = 44100;
constexpr double PI = 3.14159265358979323846;

// Input
bool isUseCamera = true;
string inputFile = "input.jpg";
string outputWav = "output.wav";
int targetResolution = 32;

// Image postprocess
bool isInvertImage = false; // false
bool isExtractEdges = true; // true
bool isApplyThreshold = false; // false
double thresholdLevel = 172; // 172

// Generator
float minFrequency = 200;  // 200
float maxFrequency = 2000; // 2000 - 6400
float numSecondsBase = 0.1;
float numSecondsTotal = 0.3;
constexpr size_t maxBufferSize = SAMPLE_RATE * 0.75;
size_t last_timecode_limit = SAMPLE_RATE * 25;


// Tools (math)


void rot(int n, int &x, int &y, int rx, int ry) {
    int t;
    if ( ry == 0 ) {
        if ( rx == 1 ) {
            x = n - 1 - x;
            y = n - 1 - y;
        }
        t = x;
        x = y;
        y = t;
    }
    return;
}


int i4_power ( int i, int j )
{
  int k, value;

  if ( j < 0 )
  {
    if ( i == 1 ) {
      value = 1;
    } else {
      value = 0;
    }
  } else if ( j == 0 or j == 1 ) {
    value = 1;
  } else {
    value = 1;
    for ( k = 1; k <= j; k++ ) {
      value = value * i;
    }
  }
  return value;
}


void d2xy(int m, int d, int &x, int &y) {
    int n, rx, ry, s;
    int t = d;

    n = i4_power ( 2, m );

    x = 0;
    y = 0;
    for ( s = 1; s < n; s = s * 2 )
    {
        rx = ( 1 & ( t / 2 ) );
        ry = ( 1 & ( t ^ rx ) );
        rot ( s, x, y, rx, ry );
        x = x + s * rx;
        y = y + s * ry;
        t = t / 4;
    }
    return;
}


bool checkPowerOf2(int num) {
    return (num & (num - 1)) == 0 && num != 0;
}


double generateSine(double f, double a, double t) {
    return sin(t / SAMPLE_RATE * 2 * PI * f) * a;
}


// ALSA control


void alsaErrorHandler(const char *file, int line, const char *function, int err, const char *fmt, ...) {
    // Ignore errors
}


void disableAlsaWarnings() {
    snd_lib_error_set_handler(alsaErrorHandler);
}


// Generation


Mat getImage(cv::VideoCapture captureSession) {

    Mat frame;

    // Get raw image
    if (isUseCamera) {

        // Capture image from camera
        if (!captureSession.isOpened()) {
            cerr << "Capture session error" << endl;
        }
        captureSession >> frame;
        if (frame.empty()) {
            cerr << "Capture error" << endl;
        }
    } else {

        // Load image by path
        frame = imread(inputFile);
        if (frame.empty()) {
            cerr << "Error opening image file!" << endl;
        }
    }

    // Convert to grayscale
    cvtColor(frame, frame, COLOR_BGR2GRAY);

    // Resize
    resize(frame, frame, Size(targetResolution, targetResolution), 0, 0, INTER_AREA);

    // Validate dimensions
    int x = frame.cols, y = frame.rows;
    if (x != y || !checkPowerOf2(x)) {
        cerr << "The image must be a square with dimensions as a power of 2." << endl;
    }

    return frame;
}


Mat postprocessImage(Mat image) {

    // Inversion
    if (isInvertImage) {
        bitwise_not(image, image);
    }

    // Edges
    if (isExtractEdges) {
        GaussianBlur(image, image, Size(5, 5), 1.5);
        Canny(image, image, 100, 150);
    }
    
    // Throshold
    if (isApplyThreshold) {
        threshold(image, image, thresholdLevel, 255, THRESH_BINARY);
    }

    return image;
}


void showImageAsPreview(Mat image) {
    // Preview
    Mat previewFrame;
    resize(image, previewFrame, Size(320, 320), 0, 0, INTER_NEAREST); // Resize
    flip(previewFrame, previewFrame, 1); // Flip
    imshow("Preview", previewFrame); // Show
    waitKey(1); // Wait
}


size_t last_timecode = 0;


vector<float> generateAudioFromImage(Mat image) {

    // Process data
    int x = image.cols, y = image.rows;

    // Combine a pixels from image to queue by their position on Hilbert curve
    vector<int> pixelQueue;
    int hx, hy;

    for (int i = 0; i < x * y; ++i) {
        d2xy(log2(x), i, hx, hy);
        pixelQueue.push_back(image.at<uchar>(hy, hx));
        // std::cout << "I: " << i << "  HX: " << hy << "  HY: " << hx << std::endl;
    }
    
    // Generate tones by pixel position in queue and it's brightness
    vector<float> outputAudio(SAMPLE_RATE * numSecondsBase, 0.0f);

    for (size_t i = 0; i < pixelQueue.size(); ++i) {

        // Calculate wave parameters for each pixel
        float position = static_cast<float>(i) / (x * y) + 0.01f;
        //float frequency = minFrequency + (maxFrequency - minFrequency) * position;
        float frequency = minFrequency + (maxFrequency - minFrequency) * (position * position);
        float amplitude = 1.0f / (x * y / 2);
        float volume = pixelQueue[i] / 255.0f;
        // std::cout << "V: " << volume << std::endl;

        // Generate audio data with target duration
        for (size_t t = 0; t < outputAudio.size(); ++t) {
            outputAudio[t] += generateSine(frequency, amplitude * volume, t + last_timecode);
        }
    }

    // Move last timecode
    last_timecode += outputAudio.size();
    //std::cout << "LTC: " << last_timecode << std::endl;
    // Apply limit
    if (last_timecode > last_timecode_limit) {
        last_timecode = 0;
    }

    // Multiply audio duration for target value
    for (int i = 0; i + 1 < numSecondsTotal / numSecondsBase; ++i) {
        outputAudio.insert(outputAudio.end(), outputAudio.begin(), outputAudio.end());
    }

    return outputAudio;
}


std::vector<float> audioBuffer;        // Main audio buffer
std::mutex audioBufferMutex;           // Mutex for buffer synchronization
std::condition_variable audioBufferCv; // Condition
std::deque<float> audioBufferBackup;   // Backup buffer to store 1 second of data
std::mutex audioBufferBackupMutex;     // Mutex for backup buffer synchronization


void writeSoundDataToBuffer(vector<float> audioData) {

    // Write data to the buffer
    {
        // Unlock mutex
        std::unique_lock<std::mutex> lock(audioBufferMutex);

        // Check buffer size, if its too large -> skip
        if (audioBuffer.size() >= static_cast<int>(maxBufferSize)) {
            return; 
        }

        // Add data to the shared buffer
        audioBuffer.insert(audioBuffer.end(), audioData.begin(), audioData.end());
        //std::cout << "Add data, buffer size: " << audioBuffer.size() << std::endl;
    }

    // Notify playback thread if waiting for data
    audioBufferCv.notify_one();
}


// Audio player


int audioCallback(const void *inputBuffer, void *outputBuffer,
                  unsigned long framesPerBuffer,
                  const PaStreamCallbackTimeInfo *timeInfo,
                  PaStreamCallbackFlags statusFlags,
                  void *userData) {
    float *out = static_cast<float *>(outputBuffer);
    size_t samplesAvailable = 0;

    {
        std::lock_guard<std::mutex> lock(audioBufferMutex);
        samplesAvailable = audioBuffer.size();
    }

    if (samplesAvailable < framesPerBuffer) {
        // Not enough data: use backup buffer or silence
        std::cout << "Low data rate from generator" << std::endl;

        std::lock_guard<std::mutex> lock(audioBufferBackupMutex);
        if (!audioBufferBackup.empty()) {
            // Repeat data from the backup buffer
            size_t backupSamples = std::min(audioBufferBackup.size(), static_cast<size_t>(framesPerBuffer));
            std::copy(audioBufferBackup.begin(), audioBufferBackup.begin() + backupSamples, out);

            // If less than framesPerBuffer, fill the remaining with silence
            std::fill(out + backupSamples, out + framesPerBuffer, 0.0f);

            // Wrap around if necessary
            if (audioBufferBackup.size() > framesPerBuffer) {
                audioBufferBackup.erase(audioBufferBackup.begin(), audioBufferBackup.begin() + backupSamples);
            }
        } else {
            // No backup: fill with silence
            std::fill(out, out + framesPerBuffer, 0.0f);
        }
    } else {
        // Copy data from the main buffer
        //std::cout << "Copy data" << std::endl;

        {
            std::lock_guard<std::mutex> lock(audioBufferMutex);
            std::copy(audioBuffer.begin(), audioBuffer.begin() + framesPerBuffer, out);
            audioBuffer.erase(audioBuffer.begin(), audioBuffer.begin() + framesPerBuffer);
        }

        // Update the backup buffer
        {
            std::lock_guard<std::mutex> lock(audioBufferBackupMutex);
            audioBufferBackup.insert(audioBufferBackup.end(), out, out + framesPerBuffer);

            // Ensure the backup buffer does not exceed 1 second of audio
            if (audioBufferBackup.size() > 44100) {
                audioBufferBackup.erase(
                    audioBufferBackup.begin(), audioBufferBackup.begin() + (audioBufferBackup.size() - 44100));
            }
        }
    }

    return paContinue;
}


// Main loop


void generator() {
    VideoCapture captureSession(0);
    while (true) {
        Mat image = getImage(captureSession);
        image = postprocessImage(image);
        showImageAsPreview(image);
        vector<float> audioData = generateAudioFromImage(image);
        writeSoundDataToBuffer(audioData);
    }
}


int main() {

    disableAlsaWarnings();


    // Init PortAudio
    Pa_Initialize();

    // Open stream
    PaStream* stream;
    Pa_OpenDefaultStream(&stream, 0, 1, paFloat32, 44100, 256, audioCallback, nullptr);

    // Launch stream
    Pa_StartStream(stream);

    // Laulnch generator
    std::thread generatorThread(generator);

    // Await for exit
    generatorThread.join();
    Pa_StopStream(stream);
    Pa_CloseStream(stream);
    Pa_Terminate();

    return 0;
}
