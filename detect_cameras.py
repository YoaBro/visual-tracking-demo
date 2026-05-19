"""
Simple helper to probe camera indices on Windows.
Run: python detect_cameras.py

It will try indices 0..8 and print which indexes open a frame.
"""
import cv2

MAX_INDEX = 8

print('Probing camera indices 0..8 (this may take a few seconds)')
available = []
for i in range(0, MAX_INDEX + 1):
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap.release()
        continue
    ret, frame = cap.read()
    cap.release()
    if ret:
        print(f'Camera index {i}: OK')
        available.append(i)
    else:
        print(f'Camera index {i}: opened but no frame')

if not available:
    print('No cameras found.')
else:
    print('\nAvailable camera indices:', available)
    print('Choose the desired index and set CAMERA_INDEX in config.py')
