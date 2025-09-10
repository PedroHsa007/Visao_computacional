#Projeto de visão computacional para identificar "pontos" no corpo
#Feito por: Pedro Henrique Oliveira de Sá
#Data: 06/09/2025

import cv2
import mediapipe as mp
import numpy as np

# Inicializar MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_holistic = mp.solutions.holistic

# Inicializar a captura de vídeo
cap = cv2.VideoCapture(0)

# Dicionário para mapear partes do corpo
body_part_names = {
    0: "Nariz", 1: "Olho esquerdo", 2: "Olho direito", 3: "Orelha esquerda", 4: "Orelha direita",
    5: "Ombro esquerdo", 6: "Ombro direito", 7: "Cotovelo esquerdo", 8: "Cotovelo direito",
    9: "Punho esquerdo", 10: "Punho direito", 11: "Quadril esquerdo", 12: "Quadril direito",
    13: "Joelho esquerdo", 14: "Joelho direito", 15: "Tornozelo esquerdo", 16: "Tornozelo direito",
    17: "Calcâneo esquerdo", 18: "Calcâneo direito", 19: "Pé esquerdo", 20: "Pé direito"
}

# Cores para diferentes partes do corpo
colors = {
    "face": (0, 255, 255),        # Amarelo
    "left_hand": (0, 0, 255),     # Vermelho
    "right_hand": (255, 0, 0),    # Azul
    "left_foot": (0, 255, 0),     # Verde
    "right_foot": (255, 0, 255),  # Magenta
    "body": (255, 255, 0)         # Ciano
}

with mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as holistic:
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignorando quadro vazio da câmera.")
            continue

        # Converter a imagem BGR para RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = holistic.process(image)

        # Desenhar anotações na imagem
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Desenhar pose, mãos e rostos
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=colors["body"], thickness=2, circle_radius=2),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=colors["body"], thickness=2))
        
        mp_drawing.draw_landmarks(
            image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=colors["left_hand"], thickness=2, circle_radius=2),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=colors["left_hand"], thickness=2))
        
        mp_drawing.draw_landmarks(
            image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=colors["right_hand"], thickness=2, circle_radius=2),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=colors["right_hand"], thickness=2))
        
        mp_drawing.draw_landmarks(
            image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=colors["face"], thickness=1, circle_radius=1),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=colors["face"], thickness=1))

        # Adicionar informações na imagem
        cv2.putText(image, "Sistema de Detecção de Partes do Corpo", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Mostrar quais partes estão sendo detectadas
        y_offset = 60
        for part, color in colors.items():
            status = "Detectado" if (
                (part == "face" and results.face_landmarks) or
                (part == "left_hand" and results.left_hand_landmarks) or
                (part == "right_hand" and results.right_hand_landmarks) or
                (part == "body" and results.pose_landmarks)
            ) else "Nao detectado"
            
            # Para os pés, verificar pontos específicos da pose
            if part == "left_foot" and results.pose_landmarks and len(results.pose_landmarks.landmark) > 27:
                status = "Detectado" if results.pose_landmarks.landmark[27].visibility > 0.5 else "Nao detectado"
            elif part == "right_foot" and results.pose_landmarks and len(results.pose_landmarks.landmark) > 28:
                status = "Detectado" if results.pose_landmarks.landmark[28].visibility > 0.5 else "Nao detectado"
            
            cv2.putText(image, f"{part}: {status}", (10, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y_offset += 30

        # Mostrar a imagem
        cv2.imshow('Detecção de Partes do Corpo', image)
        
        # Sair ao pressionar 'q'
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()