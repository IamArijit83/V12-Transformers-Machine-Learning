################################################ Madoc Health Assignment ####################################################################
################################################ Biometric(Face Recognition) ################################################################
############################################# By Arijit Patra (MCA (Hons. AI-ML)) ###########################################################
############################################### Registration Number- 12408191 ###############################################################

################################ Headers ##########################################
import streamlit as st
import pandas as pd
import cv2
import numpy as np
import os
import time
from datetime import datetime
from deepface import DeepFace

####### 1. Page Config #######
st.set_page_config(page_title="SafeEntry Pro", page_icon="🛡️", layout="wide")

####### 2. Setting up Session State #######
# We need two separate keys: one for the Attendance camera, one for Registration
if 'camera_key' not in st.session_state:
    st.session_state.camera_key = 0

if 'reg_key' not in st.session_state:
    st.session_state.reg_key = 0

####### 3. Setting Up Folders #######
if not os.path.exists("database"):
    os.makedirs("database")

if not os.path.exists("attendance.csv"):
    with open("attendance.csv", "w") as f:
        f.write("ID,Name,Role,Timestamp,Action\n")

####### 4. Fixing the Lighting (CLAHE) #######
def enhance_image(image_path):
    img = cv2.imread(image_path)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    cv2.imwrite("temp_enhanced.jpg", enhanced_img)
    return "temp_enhanced.jpg"

####### 5. Checking the Status #######
def check_last_action(user_id):
    try:
        df = pd.read_csv("attendance.csv", dtype={'ID': str}) 
        user_logs = df[df["ID"] == str(user_id)]
        if user_logs.empty:
            return None 
        return user_logs.iloc[-1]["Action"]
    except Exception:
        return None

####### 6. Checking if ID already exists #######
def check_id_exists(new_id):
    for file in os.listdir("database"):
        if file.startswith(f"{new_id}_"):
            return True
    return False

####### 7. Marking Attendance #######
def mark_attendance(user_id, name, role, action):
    with open("attendance.csv", "a") as f:
        now = datetime.now()
        dt_string = now.strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"{user_id},{name},{role},{dt_string},{action}\n")

####### 8. Front-End UI Layout #######
st.title("Madoc🛡️: Biometric Attendance")

menu = ["Punch Attendance", "Register User", "View Logs", "Project Documentation"]
choice = st.sidebar.radio("Navigation", menu)

############################# --- SECTION 1: PUNCH IN / PUNCH OUT --- #############################

if choice == "Punch Attendance":
    st.subheader("📸 Biometric Verification")
    
    action_type = st.radio("Select Action:", ["Punch IN", "Punch OUT"], horizontal=True)
    
    # Dynamic Key resets camera on switching actions
    unique_key = f"camera_{st.session_state.camera_key}_{action_type}"
    
    img_file = st.camera_input("Look at the camera", key=unique_key)

    if img_file is not None:
        if st.button(f"Confirm {action_type}"):
            with open("temp.jpg", "wb") as f:
                f.write(img_file.getbuffer())
            
            clean_img_path = enhance_image("temp.jpg")

            try:
                with st.spinner("Verifying Identity & Rules..."):
                    
                    ####### A. Recognition #######
                    dfs = DeepFace.find(
                        img_path=clean_img_path, 
                        db_path="database", 
                        model_name="VGG-Face", 
                        enforce_detection=False
                    )

                    if len(dfs) > 0 and not dfs[0].empty:
                        matched_path = dfs[0].iloc[0]['identity']
                        filename = os.path.basename(matched_path)
                        
                        try:
                            user_id = filename.split("_")[0]
                            name = filename.split("_")[1]
                            role = filename.split("_")[2].split(".")[0]
                        except IndexError:
                            st.error("❌ Error: Database filename format incorrect.")
                            st.stop()

                        ####### B. Logic Check #######
                        last_status = check_last_action(user_id)
                        
                        if action_type == "Punch IN" and last_status == "Punch IN":
                            st.warning(f"⚠️ User {user_id} ({name}), you are ALREADY punched in!")
                        
                        elif action_type == "Punch OUT" and (last_status == "Punch OUT" or last_status is None):
                            st.warning(f"⚠️ User {user_id} ({name}), you are ALREADY checked out.")
                        
                        ####### C. Anti-Spoofing #######
                        else:
                            face_objs = DeepFace.extract_faces(
                                img_path=clean_img_path, 
                                anti_spoofing=True
                            )
                            is_real = face_objs[0].get("is_real", True)
                            
                            if not is_real:
                                st.error("⚠️ SPOOF DETECTED! Fake face prevented.")
                            else:
                                mark_attendance(user_id, name, role, action_type)
                                
                                ####### Success Message #######
                                msg_placeholder = st.empty()
                                msg_placeholder.success(f"✅ Success! {name} (ID: {user_id}) marked {action_type}")
                                time.sleep(3)
                                
                                ####### Reset Camera #######
                                st.session_state.camera_key += 1
                                st.rerun()

                    else:
                        st.error("❌ User not found. Please register first.")
            
            except ValueError:
                st.warning("No face detected. Please align properly.")
            except Exception as e:
                st.error(f"An error occurred: {e}")

############################# --- SECTION 2: REGISTER USER (SECURED) --- #############################
elif choice == "Register User":
    st.subheader("📝 New Registration")
    
    c1, c2, c3 = st.columns(3)
    new_id = c1.text_input("ID (5-Digit)", max_chars=5)
    name = c2.text_input("Name")
    role = c3.selectbox("Role", ["Employee", "Manager", "Visitor"])
    
    # DYNAMIC REGISTRATION CAMERA
    # We use a separate key 'reg_key' to force the camera to reset after saving
    reg_img = st.camera_input("Register Face", key=f"reg_cam_{st.session_state.reg_key}")
    
    if st.button("Save Profile"):

        ####### Validation checking #######
        if not new_id or not new_id.isdigit() or len(new_id) != 5:
            st.error("❌ ID must be exactly 5 numeric digits (e.g., 10001).")
        
        elif not name:
            st.error("❌ Please enter a name.")
            
        elif check_id_exists(new_id):
            st.error(f"❌ ID {new_id} is already registered! Please use a unique ID.")
            
        elif reg_img:
            ####### Saving format: ID_Name_Role.jpg
            filename = f"database/{new_id}_{name}_{role}.jpg"
            with open(filename, "wb") as f:
                f.write(reg_img.getbuffer())
            
            ####### SUCCESS FEEDBACK LOOP #######
            msg_placeholder = st.empty()
            msg_placeholder.success(f"✅ Registration Successful! Added {name} (ID: {new_id})")
            
            # Wait 3 seconds so user sees the message and gets confirmed
            time.sleep(3)
            
            # Increment key to destroy the old camera instance (for auto refresh)
            st.session_state.reg_key += 1
            
            # Rerunning to clear text inputs and show fresh camera
            st.rerun()
        else:
            st.error("❌ Please capture a photo.")

############################# --- SECTION 3: LOGS --- #############################
elif choice == "View Logs":
    st.subheader("📊 Activity Log")
    if os.path.exists("attendance.csv"):
        df = pd.read_csv("attendance.csv", dtype={'ID': str})
        st.dataframe(df.style.highlight_max(axis=0))


############################# --- SECTION 4: A Minimal Project Documentation (For Easy documentation Access) --- #############################

elif choice == "Project Documentation":
    st.subheader("📘 Project Deliverables")
    st.markdown("""
    ### 1. Model & Approach
    * **Model:** VGG-Face (Visual Geometry Group).
    * **Approach:** One-Shot Learning with ID-based verification.
    
    ### 2. Security Features
    * **Anti-Spoofing:** Texture/Depth analysis.
    * **Duplicate Prevention:** 5-Digit Unique ID enforcement.
    * **State Management:** Prevents double check-ins/check-outs.
    * **Session Security:** Camera and Inputs auto-reset after every transaction.
    """)