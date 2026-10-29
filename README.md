# Doorlog — Smart Attendance System (AI Face Recognition) | FYP

Doorlog is an AI-powered smart attendance system designed to recognize registered students, teachers, and staff members from different face angles. It records attendance, including check-in and check-out times, and generates alerts with photos when an unknown person or someone concealing their face enters the monitored area.

The project also includes the original **Video Door Counter** feature in the **Video Analysis** tab, where recognized people can be identified by name.

## 1. Initial Setup (One Time)

```bash
cd backend
pip install -r requirements.txt
python download_models.py
```

The `download_models.py` script downloads the required face recognition models. An internet connection is required, and the download is approximately 37 MB.

## 2. Run the Application

Run these commands whenever you want to start the application:

```bash
cd backend
uvicorn main:app
```

Open the application in your browser:

**http://localhost:8000**

Use `localhost` when capturing photos through the computer's webcam.

## 3. How to Use the System

### 3.1 People Tab — Register People

For each person, enter their name, roll number, and role, then provide three photos:

- **Front:** Facing directly toward the camera.
- **Left:** Turn the head approximately 30–45 degrees to the left.
- **Right:** Turn the head approximately 30–45 degrees to the right.

You can select photos from your device or capture them using the **Use Webcam** option.

For better recognition, use good lighting and avoid face coverings. Additional photos can be uploaded later using **Add Photos**.

### 3.2 Live Camera Tab — Monitor Entry and Exit

Select the appropriate camera source:

- Laptop webcam (`0`)
- USB camera (`1`)
- CCTV camera using an RTSP address
- Uploaded video for testing

Choose one of the following operating modes:

- **Entry Camera:** Place the camera at the entrance. Recognized people are marked as **CHECK IN**.
- **Exit Camera:** Monitor people leaving the building. Recognized people are marked as **CHECK OUT**.
- **Door Line:** Use a single camera to monitor both entry and exit based on the direction in which people cross a configured line.

Click **Start** to begin monitoring.

Detection colors indicate the following:

- **Green:** Registered person — displays their name and roll number.
- **Red:** Unknown person.
- **Orange:** Face-hidden detection.

### 3.3 Attendance Tab — View Attendance Records

Select a date to view attendance records for registered people, including:

- Present or absent status
- Check-in time
- Check-out time
- Total time spent inside

You can download attendance records as a CSV file.

If the camera misses an event, authorized staff can use **Mark In** or **Mark Out** to record attendance manually.

### 3.4 Alerts Tab — Review Security Alerts

The system records alerts for unknown people and face-hidden detections. Each alert may include:

- A snapshot showing the person and face, when available
- Detection timestamp
- Camera information

Staff can review each alert and mark it as **Checked by Staff**.

If an unknown person is recognized later, the system displays information such as **Later Recognised As...**, where supported by the implementation.

## 3.5 Test with a Video Before Connecting CCTV

### Step 1: Upload a Test Video

Open the **Video Analysis** tab and upload a video showing people entering or leaving a doorway.

### Step 2: Start Video-Based Monitoring

Open the **Live Camera** tab, select **Uploaded Video**, and click **Start**.

The uploaded video is processed as a camera source, and attendance events are recorded using the application's current date and time.

### Step 3: Connect CCTV Later

Before using CCTV, ensure that the computer running Doorlog and the DVR/NVR are connected to the same network.

1. Open the **Live Camera** tab.
2. Select **CCTV Camera**.
3. Choose the camera brand: Hikvision, Dahua, CP Plus, or Other.
4. Enter the DVR/NVR IP address, username, password, and camera channel.
5. For the Other option, enter the complete RTSP address.
6. Click **Test Connection** and verify that the camera feed appears.
7. If the connection succeeds, click **Start Camera**.
8. Start with the **Sub Stream** for better performance. If faces are too small, try the **Main Stream**.

The application is designed to retry the connection periodically if the network or DVR connection is interrupted, provided this behavior is supported by the running implementation.

**Finding the DVR IP address:** Check the DVR's Network settings under its menu. You can also use the manufacturer's discovery software, such as Hikvision SADP or Dahua ConfigTool.

## 4. How the System Works

Doorlog combines object detection, tracking, and face recognition:

1. **YOLOv8:** Detects people in video frames.
2. **ByteTrack:** Assigns tracking IDs to detected people.
3. **YuNet:** Detects faces approximately every 0.25 seconds.
4. **SFace:** Generates a 128-dimensional face representation for comparison.
5. **Face Matching:** Compares the detected face with registered face representations, including front, left, right, and mirrored images.
6. **Recognition Confirmation:** Requires two matching votes before assigning an identity, helping reduce incorrect matches.

Adding a person is an enrollment process; the model does not need to be retrained from scratch.

### Unknown Person Detection

If a visible face cannot be matched to a registered person for approximately three seconds, the system can generate an **UNKNOWN** alert.

### Face-Hidden Detection

If a person approaches the camera but their face remains undetected for approximately four seconds, the system can generate a **FACE HIDDEN** alert, subject to the configured detection logic.

### Data Storage

The application's local data is stored in:

```text
backend/data/
```

This directory may contain:

- Registered people's photos
- `attendance.db` — SQLite attendance database
- Alert snapshots

Back up this directory regularly when running the application locally.

## 5. Camera Placement Tips

Camera placement significantly affects recognition accuracy.

- Mount the camera approximately **1.8–2.2 metres** above the floor.
- Position it facing people as they enter, with a slight downward angle.
- Aim for a face size of at least approximately **80 pixels** in the frame.
- Avoid strong backlighting from windows or direct sunlight.
- Ensure that faces are sufficiently visible and illuminated.
- In Entry Camera mode, a person walking away from the camera may not trigger a face-hidden alert because their face is not expected to remain visible.

Actual performance depends on camera quality, lighting, distance, movement, and the recognition threshold.

## 6. Configuration and Tuning

Configuration settings are located in:

```text
backend/config.py
```

Within `InstitutionConfig`, the following parameters can be adjusted if these names and defaults match your current implementation:

| Parameter | Example Default | Purpose |
|---|---:|---|
| `recog_thr` | `0.40` | Face recognition threshold |
| `votes_needed` | `2` | Number of matching votes required |
| `unknown_after_seconds` | `3` | Time before an unknown-person alert |
| `hidden_after_seconds` | `4` | Time before a face-hidden alert |
| `log_cooldown_seconds` | `60` | Cooldown between attendance log events |

**Tuning guidance:**

- If incorrect identities are assigned, try increasing `recog_thr`, for example to `0.45`.
- If registered people are frequently missed, carefully test a lower value, such as `0.37`.
- Always evaluate changes using representative test videos. Lower thresholds can increase false matches, while higher thresholds can increase missed recognitions.

## 7. Run the Tests

From the `backend` directory, run:

```bash
python -m tests.test_logic
python -m tests.test_pipeline
python -m tests.test_institution
```

These tests help verify the application's logic and processing pipeline.

## 8. Important Limitations

- The system must be tested on the target machine with a real webcam and the actual face recognition models.
- Passing tests with simulated cameras or faces does not guarantee reliable real-world recognition.
- Recognition accuracy depends on image quality, face visibility, camera placement, lighting, and the selected thresholds.
- A cloud-hosted application may not be able to access a laptop's webcam or a private-network CCTV feed directly. Test local camera access on the computer running the backend.
- Free hosting environments may have memory, processing, uptime, or persistent-storage limitations. Confirm that attendance records and uploaded images are stored persistently before relying on them.

## 9. Privacy and Data Security

Face images and biometric representations are sensitive personal data.

- Obtain informed permission before registering people.
- For children, obtain the appropriate consent from their parents or guardians.
- Restrict access to registered face photos, attendance records, and alert snapshots.
- Keep the `backend/data/` directory private and secure.
- Do not commit real face images, attendance databases, credentials, or private CCTV details to a public GitHub repository.
- Use sample videos and dummy data for public demonstrations.
- Establish appropriate retention and deletion procedures for personal data.

---

**Project:** Doorlog — Smart Attendance System (AI Face Recognition)

**Purpose:** AI-based attendance tracking and doorway monitoring for educational institutions and workplaces.

**Project Type:** Final Year Project (FYP)
