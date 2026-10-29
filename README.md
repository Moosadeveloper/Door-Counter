# Doorlog - Smart Attendance (AI Face Recognition) - FYP

Camera registered logon (students / teachers / staff) ko **kisi bhi angle se** pehchanta hai,
unki **attendance (Present + kis time andar aaya / bahar gaya)** lagata hai, aur jo banda
**registered nahi** ya **chehra chupa ke** andar aaye uska **ALERT + photo** save karta hai.
Purana "video door counter" bhi saath hai (Video analysis tab) - ab usme bhi naam dikhte hain.

## 1. Pehli dafa setup (sirf ek baar)
```
cd backend
pip install -r requirements.txt
python download_models.py      # face models download (internet chahiye, ~37 MB)
```

## 2. Chalana (har dafa)
```
cd backend
uvicorn main:app
```
Browser mein kholo: **http://localhost:8000**   (webcam se photo lene ke liye localhost hi use karo)

## 3. Istemal ka tareeqa
1. **People tab** - har bande ka naam, roll no, role likho aur 3 photos do:
   - **Front** (seedha camera ki taraf)
   - **Left side** (sar ~30-45 degree left - poora side-view nahi)
   - **Right side** (sar ~30-45 degree right)
   Photo file choose karo ya "Use webcam" se wahin khainch lo. Achhi roshni, mask/chashma nahi.
   Baad mein "Add photos" se aur photos add kar sakte ho (zyada photos = behtar pehchan).
2. **Live camera tab** - camera chuno (laptop webcam `0`, USB camera `1`, CCTV ka `rtsp://...` address,
   ya testing ke liye upload ki hui video) aur mode chuno:
   - **Entry camera**: gate par, andar aane walon ke chehre ki taraf -> pehchana = CHECK IN
   - **Exit camera**: bahar jane walon ki taraf -> pehchana = CHECK OUT
   - **Door line**: ek camera door ke dono taraf dekhta hai -> line draw karo, direction se IN / OUT
   Start dabao. Screen par: hara = registered (naam + roll no), laal = UNKNOWN, orange = FACE HIDDEN.
3. **Attendance tab** - din chuno: har registered banda Present / Absent, aane ka time, jane ka time,
   kitni der andar raha. CSV download. Camera se miss ho jaye to "Mark in / Mark out" (manual).
4. **Alerts tab** - unknown / face-hidden logon ki photo (poora banda + chehra), time, camera.
   Staff check kar ke "Checked by staff" tick kare. Agar wohi banda baad mein pehchana gaya to
   "Later recognised as ..." likha aata hai.

## 3b. Pehle VIDEO se test, phir CCTV live
**Abhi (video se test):**
1. Video analysis tab mein apni video upload karo (gate / door ki recording jisme log aate hon).
2. Live camera tab -> "Uploaded video" chuno -> Start. Video asli camera ki tarah normal speed par chalegi
   aur attendance aaj ke time ke saath lagegi.

**Baad mein (CCTV live):**
1. Computer aur DVR/NVR **ek hi network** (router / Wi-Fi) par hon.
2. Live camera tab -> "CCTV camera" -> brand chuno (Hikvision / Dahua / CP Plus, warna "Other" aur poora RTSP address).
3. DVR ka IP address, user name (aksar `admin`), password, aur channel (kaun sa camera: 1, 2, 3...) daalo.
4. **Test connection** dabao - camera ki picture aa jaye to **Start camera**.
5. Quality "Sub stream" rakho (tez chalta hai). Chehre chhote aa rahe hon to "Main stream" try karo.
6. Network / DVR ka connection toot jaye to system khud har kuch second baad dobara jodta hai.
   Password browser mein save nahi hota; baqi settings yaad rehti hain.

DVR ka IP pata na ho: DVR ki screen par Menu -> Network mein likha hota hai (jaise 192.168.1.64),
ya Hikvision ka "SADP" / Dahua ka "ConfigTool" software use karo.

## 4. Andar kya ho raha hai
- YOLOv8 log dhoondta hai, ByteTrack har bande ko ID deta hai.
- Har ~0.25 sec bande ka chehra (YuNet) -> 128 numbers ka fingerprint (SFace).
- Ye fingerprint registered logon ki **har photo** (front/left/right + unki mirror copy) se compare hota hai.
  **2 dafa match** ho to naam lagta hai (ek dafa ke galat match se attendance na lage).
- "Training" = **enrollment**: naye bande ko add karna 1 second ka kaam hai, model dobara train nahi karna padta.
- Chehra dikha lekin kisi se match nahi (3 sec) -> **UNKNOWN** alert.
  Banda camera ki taraf aaya lekin chehra kabhi nazar nahi aaya (4 sec) -> **FACE HIDDEN** alert.
- Data: `backend/data/` (registry photos + `attendance.db` SQLite + alert snapshots). Backup ke liye ye folder copy karo.

## 5. Camera lagane ke tips (sab se zaroori)
- Camera chehre ki height par (~1.8-2.2 m), aane walon ke **saamne**, halka sa neeche jhuka hua.
- Chehra frame mein kam se kam ~80 px ka ho. Door se bahut door camera = pehchan kamzor.
- Peeche se tez roshni (khidki/dhoop) na ho - chehra kaala aata hai.
- Entry mode mein jo camera se door ja raha ho (peeth dikha raha ho) us par face-hidden alert nahi aata.

## 6. Tuning (backend/config.py -> InstitutionConfig)
- `recog_thr` (0.40): galat naam lag raha ho to barhao (0.45); sahi banda pehchana na jaye to kam karo (0.37).
- `votes_needed` (2), `unknown_after_seconds` (3), `hidden_after_seconds` (4), `log_cooldown_seconds` (60).

## 7. Tests (bina camera ke)
```
cd backend
python -m tests.test_logic
python -m tests.test_pipeline
python -m tests.test_institution
```

## Note
- Real webcam / real face models ke saath apni machine par test karna zaroori hai
  (logic fake camera + fake faces se test hua hai).
- Chehron ka data sensitive hai: register karne se pehle logon (aur bachon ke case mein walidain) ki
  ijazat lo, aur `data/` folder mehfooz rakho.
