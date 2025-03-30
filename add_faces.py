import face_recognition
import os
import pickle


KNOWN_FACES_DIR = "known_faces"

def encode_faces():
    encodings = []
    names = []

    for filename in os.listdir(KNOWN_FACES_DIR):
        image_path = os.path.join(KNOWN_FACES_DIR, filename)
        image = face_recognition.load_image_file(image_path)
        encoding = face_recognition.face_encodings(image)

        if encoding:  
            encodings.append(encoding[0])
            names.append(os.path.splitext(filename)[0])

    
    with open("encodings.pickle", "wb") as f:
        pickle.dump((encodings, names), f)

    print(f"✅ Encoded {len(encodings)} faces.")


if __name__ == "__main__":
    encode_faces()
