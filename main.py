import pickle

import cv2
import face_recognition
from PIL import Image, ImageDraw

count = 0


def face_rec():
    curt_face_img = face_recognition.load_image_file('img/KurtCobain.jpg')
    curt_face_location = face_recognition.face_locations(curt_face_img)

    team_face_img = face_recognition.load_image_file('../img/team.png')
    team_face_location = face_recognition.face_locations(team_face_img)

    print(curt_face_location)
    print(team_face_location)
    print(f"Found {len(curt_face_location)} face(s) in this image")
    print(f"Found {len(team_face_location)} face(s) in this image")

    pil_img1 = Image.fromarray(curt_face_img)
    draw1 = ImageDraw.Draw(pil_img1)

    for (top, right, bottom, left) in curt_face_location:
        draw1.rectangle(((top, right), (bottom, left)), outline=(0, 255, 0), width=10)

    del draw1
    pil_img1.save("img/new_curt.jpg")

    pil_img2 = Image.fromarray(team_face_img)
    draw2 = ImageDraw.Draw(pil_img2)

    for (top, right, bottom, left) in team_face_location:
        draw2.rectangle(((top, right), (bottom, left)), outline=(0, 255, 0), width=10)

    del draw2
    pil_img2.save("img/new_team.jpg")


def extracting_faces(img_path):
    global count
    faces = face_recognition.load_image_file(img_path)
    faces_locations = face_recognition.face_locations(faces)

    for faces_location in faces_locations:
        top, right, bottom, left = faces_location

        face_img = faces[top:bottom, left:right]
        pil_img = Image.fromarray(face_img)
        pil_img.save(f"img/TeamFaces/{count}face_img.jpg")
        pil_img.save(f"img/TeamFaces/{count}face_img.jpg")
        pil_img.save(f"img/TeamFaces/{count}face_img.jpg")
        pil_img.save(f"img/TeamFaces/{count}face_img.jpg")
        pil_img.save(f"img/TeamFaces/{count}face_img.jpg")
        pil_img.save(f"img/TeamFaces/{count}face_img.jpg")
        pil_img.save(f"img/TeamFaces/{count}face_img.jpg")
        pil_img.save(f"img/TeamFaces/{count}face_img.jpg")
        count += 1

    return f"Found {count} face(s) in this photo"


def compare_faces(img1_path, img2_path):
    img1 = face_recognition.load_image_file(img1_path)
    img1_encodings = face_recognition.face_encodings(img1)[0]
    # print(img1_encodings)

    img2 = face_recognition.load_image_file(img2_path)
    img2_encodings = face_recognition.face_encodings(img2)[0]

    result = face_recognition.compare_faces([img1_encodings], img2_encodings)
    print(result)


def detect_person_in_video():
    data = pickle.loads(open("Slava_encodings.pickle", "rb").read())
    video = cv2.VideoCapture(0)

    while True:
        ret, image = video.read()

        locations = face_recognition.face_locations(image, model="cnn")
        encodings = face_recognition.face_encodings(image, locations)

        for (face_encodings, face_location) in zip(encodings, locations):
            result = face_recognition.compare_faces(data["encodings"], face_encodings)
            match = None

            if True in result:
                match = data["name"]
                print(f'Match found! {match}')
            else:
                print("ALERT!")

            left_top = (face_location[3], face_location[0])
            right_bottom = (face_location[1], face_location[2])
            color = [0, 255, 0]
            cv2.rectangle(image, left_top, right_bottom, color)

        cv2.imshow("detect_person_in_video is running", image)

        k = cv2.waitKey(20)
        if k == ord('q'):
            print("Q pressed and program has been killed")
            break


def main():
    # face_rec()
    # print(extracting_faces('../img/Патрикейтман.jpg'))
    # print(extracting_faces('../img/Патрик бейтман.png'))
    # compare_faces("../img/Патрик бейтман.png", 'img/Патрикейтман.jpg')
    detect_person_in_video()


if __name__ == "__main__":
    main()
