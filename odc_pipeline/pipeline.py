import os
import io
import json
import smtplib
import zipfile
import torch
import numpy as np
import cv2
import pandas as pd
import imghdr
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email import encoders
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.oauth2.service_account import Credentials
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# -------------------------
# CONFIG
# -------------------------
EXCEL_FILE = "OPEN_DATA.xlsx.xlsx"
SHEET_NAME = "Sheet4"
DOWNLOAD_DIR = "downloads"
SERVICE_ACCOUNT_FILE = "drivedownloader-468416-3faca8fec2cc.json"
EMAIL_ADDRESS = "ssgprasunamba@gmail.com"
EMAIL_PASSWORD = "bsut anar ktfw gmdf"  # Google App Password
SAM_MODEL_CHECKPOINT = "sam_vit_h_4b8939(1).pth"
BATCH_SIZE = 50
LOCK_FILE = "pipeline.lock"

# -------------------------
# GLOBAL LOCK
# -------------------------
if os.path.exists(LOCK_FILE):
    print("⚠️ Another instance is already running. Exiting.")
    exit(0)
open(LOCK_FILE, 'w').close()

try:
    # -------------------------
    # LOAD EXCEL
    # -------------------------
    df = pd.read_excel(EXCEL_FILE, sheet_name=SHEET_NAME)
    if 'SegmentationAndMail' not in df.columns:
        df['SegmentationAndMail'] = ""
    if 'Mail_Sent' not in df.columns:
        df['Mail_Sent'] = "No"
    records = df.to_dict(orient="records")

    # -------------------------
    # GOOGLE DRIVE SETUP
    # -------------------------
    SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
    credentials = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    drive_service = build('drive', 'v3', credentials=credentials)

    def list_files_in_folder(folder_id):
        files = []
        page_token = None
        while True:
            response = drive_service.files().list(
                q=f"'{folder_id}' in parents and trashed=false",
                spaces='drive',
                fields='nextPageToken, files(id, name, mimeType)',
                pageToken=page_token
            ).execute()
            files.extend(response.get('files', []))
            page_token = response.get('nextPageToken', None)
            if not page_token:
                break
        return files

    def download_folder(folder_id, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        seen_names = set()
        files = list_files_in_folder(folder_id)
        for file in files:
            name, mime = file['name'], file['mimeType']
            if mime == 'application/vnd.google-apps.folder':
                download_folder(file['id'], os.path.join(output_dir, name))
                continue
            if not any(mime.startswith(x) for x in ("image/jpeg", "image/png", "image/jpg", "image/bmp", "image/tiff")):
                continue
            _, ext = os.path.splitext(name)
            if not ext:
                ext = ".jpg" if "jpeg" in mime else ".png"
                name += ext
            base, ext = os.path.splitext(name)
            count = 1
            while name in seen_names or os.path.exists(os.path.join(output_dir, name)):
                name = f"{base}_{count}{ext}"
                count += 1
            seen_names.add(name)
            file_path = os.path.join(output_dir, name)
            try:
                request = drive_service.files().get_media(fileId=file['id'])
                with io.FileIO(file_path, 'wb') as fh:
                    downloader = MediaIoBaseDownload(fh, request)
                    done = False
                    while not done:
                        _, done = downloader.next_chunk()
                if imghdr.what(file_path) is None:
                    os.remove(file_path)
            except Exception as e:
                print(f"Failed {name}: {e}")

> 2300080123_MNSSGPrasunamba:
# -------------------------
    # SAM SEGMENTATION
    # -------------------------
    def generate_image_json(image_dir, output_json):
        image_data = {}
        for f in os.listdir(image_dir):
            path = os.path.join(image_dir, f)
            if os.path.isfile(path) and f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
                size = os.path.getsize(path)
                key = f"{f}{size}"
                image_data[key] = {"filename": f, "size": size, "regions": [], "file_attributes": {}}
        with open(output_json, "w") as j:
            json.dump(image_data, j, indent=4)

    def segment_folder(image_dir, model_path, output_json, batch_size=BATCH_SIZE):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        sam = sam_model_registry["vit_h"](checkpoint=model_path)
        sam.to(device=device)

        mask_generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=32,
            pred_iou_thresh=0.9,
            stability_score_thresh=0.95,
            min_mask_region_area=500,
            box_nms_thresh=0.5
        )

        with open(output_json, "r") as json_file:
            image_data = json.load(json_file)

        images = [f for f in os.listdir(image_dir) if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff"))]

        for start in range(0, len(images), batch_size):
            batch = images[start:start + batch_size]
            print(f"Processing batch {start // batch_size + 1}: {len(batch)} images")

            with torch.no_grad():
                for filename in batch:
                    file_path = os.path.join(image_dir, filename)
                    if not os.path.isfile(file_path):
                        continue
                    file_size = os.path.getsize(file_path)
                    new_key = f"{filename}{file_size}"

                    image = cv2.imread(file_path)
                    if image is None:
                        continue
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                    masks = mask_generator.generate(image)
                    print(f"Generated {len(masks)} masks for {filename}")

                    if not masks:
                        image_data[new_key] = {"filename": filename, "size": file_size, "regions": [], "file_attributes": {}}
                        continue

                    masks = sorted(masks, key=lambda x: x["area"], reverse=True)
                    cumulative_mask = np.zeros_like(masks[0]["segmentation"], dtype=np.uint8)
                    regions = []

                    for mask in masks:
                        segmentation = mask["segmentation"].astype(np.uint8)
                        non_overlap = cv2.bitwise_and(segmentation, cv2.bitwise_not(cumulative_mask))
                        if np.count_nonzero(non_overlap) == 0:
                            continue
                        cumulative_mask = cv2.bitwise_or(cumulative_mask, non_overlap)
                        contours, _ = cv2.findContours(non_overlap, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        for contour in contours:
                            contour = contour.squeeze()
                            if contour.ndim == 1:
                                contour = contour.reshape(-1, 2)
                            if len(contour) < 3:
                                continue
                            area = cv2.contourArea(contour)
                            if area < 3000:
                                continue
                            all_points_x = contour[:, 0].tolist()
                            all_points_y = contour[:, 1].tolist()
                            regions.append({
                                "shape_attributes": {

> 2300080123_MNSSGPrasunamba:
"name": "polyline",
                                    "all_points_x": all_points_x,
                                    "all_points_y": all_points_y
                                },
                                "region_attributes": {}
                            })
                    image_data[new_key] = {"filename": filename, "size": file_size, "regions": regions, "file_attributes": {}}
                    del image, masks
                    torch.cuda.empty_cache()
            with open(output_json, "w") as json_file:
                json.dump(image_data, json_file, indent=4)
                print(f"✅ Batch {start // batch_size + 1} saved")
        print(f"✅ All batches processed — updated JSON: {output_json}")

    # -------------------------
    # ZIP FILE
    # -------------------------
    def zip_file(input_file):
        zip_path = input_file.replace(".json", ".zip")
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as z:
            z.write(input_file, arcname=os.path.basename(input_file))
        return zip_path

    # -------------------------
    # EMAIL FUNCTION
    # -------------------------
    def send_email(recipient, subject, body, attachments=[]):
        try:
            msg = MIMEMultipart()
            msg['From'] = EMAIL_ADDRESS
            msg['To'] = recipient
            msg['Subject'] = subject
            msg.attach(MIMEText(body, "plain"))
            for f in attachments:
                if not os.path.exists(f):
                    continue
                part = MIMEBase('application', 'octet-stream')
                with open(f, 'rb') as file:
                    part.set_payload(file.read())
                encoders.encode_base64(part)
                part.add_header('Content-Disposition', f'attachment; filename={os.path.basename(f)}')
                msg.attach(part)
            import ssl
            context = ssl.create_default_context()
            with smtplib.SMTP('smtp.gmail.com', 587) as server:
                server.starttls(context=context)
                server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
                server.send_message(msg)
            print(f"✅ Email sent to {recipient}")
            return True
        except Exception as e:
            print(f"❌ Email failed to {recipient}: {e}")
            return False

    # -------------------------
    # MAIN PIPELINE
    # -------------------------
    ATTRIBUTE_FILE = "Image_File_Attributes.json"
    VIDEO_LINK = "https://drive.google.com/drive/folders/1-joZa2HoDX7d1lDoREM5oWHcpoUci-rm?usp=sharing"

    for idx, row in df.iterrows():
        record = row.to_dict()
        drive_link = record.get('DriveLink')
        recipient_email = str(record.get('Email', '')).strip()
        recipient_name = str(record.get('Name', '')).strip() or (recipient_email.split("@")[0] if recipient_email else "User")
        status = str(record.get('SegmentationAndMail', '')).lower()
        mail_status = str(record.get('Mail_Sent', '')).lower()

        # 🧠 Skip fully completed rows
        if status == "completed" and mail_status == "yes":
            print(f"⏭ Skipping {recipient_email}: already processed and mailed.")
            continue

        if not drive_link or drive_link != drive_link:
            print(f"⚠️ Skipping row {idx}: no DriveLink")
            df.at[idx, 'SegmentationAndMail'] = "NoDriveLink"
            continue
        if not recipient_email:
            print(f"⚠️ Skipping row {idx}: no Email")
            df.at[idx, 'SegmentationAndMail'] = "NoEmail"
            continue

        try:
            folder_id = drive_link.split("/folders/")[1].split("?")[0]
            folder_name = os.path.join(DOWNLOAD_DIR, recipient_email.replace("@", "_at_").replace(".", "_dot_"))

> 2300080123_MNSSGPrasunamba:
# 📂 Skip download if folder exists
            if os.path.exists(folder_name) and os.listdir(folder_name):
                print(f"⏭ Skipping download for {recipient_email}, folder already exists.")
            else:
                print(f"⬇️ Downloading images.......")
                download_folder(folder_id, folder_name)
                print(f"✅ Images downloaded ")

            # 🧠 Skip segmentation if JSON exists
            json_path = os.path.join(folder_name, "segmented_metadata.json")
            if os.path.exists(json_path):
                print(f"⏭ Skipping segmentation for {recipient_email}, JSON already exists.")
            else:
                generate_image_json(folder_name, json_path)
                segment_folder(folder_name, SAM_MODEL_CHECKPOINT, json_path, BATCH_SIZE)

            zip_path = zip_file(json_path)

            # 📧 Send mail only if not already sent
            if mail_status == "yes":
                print(f"⏭ Skipping email for {recipient_email}, already sent.")
            else:
                email_body = f"""Hi {recipient_name},

Your dataset submission has been successfully processed!

Your Segmented File and Attribute File are attached below.

To help you get through the remaining annotation process, you can watch a short tutorial here: {VIDEO_LINK}
Website shown in video: https://annotate.officialstatistics.org/

After completion, get it verified with us at C424 to obtain your marks.

Thanks,
Team ODC
"""
                if send_email(recipient_email, "Segmentation Completed", email_body, [zip_path, ATTRIBUTE_FILE]):
                    df.at[idx, 'Mail_Sent'] = "Yes"
                    df.at[idx, 'SegmentationAndMail'] = "Completed"

        except Exception as e:
            print(f"❌ Error processing {recipient_email}: {e}")
            df.at[idx, 'SegmentationAndMail'] = "Error"

        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print("🧹 GPU cache cleared")

        # Save progress after every user
        df.to_excel(EXCEL_FILE, index=False)

    print("✅ Pipeline finished, Excel updated!")

finally:
    if os.path.exists(LOCK_FILE):
        os.remove(LOCK_FILE)
        print("🔓 Lock released.")
