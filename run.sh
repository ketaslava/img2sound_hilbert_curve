rm dist/main
g++ src/main.cpp -o dist/main -linput `pkg-config --cflags --libs opencv4` -lportaudio -lasound
./dist/main
