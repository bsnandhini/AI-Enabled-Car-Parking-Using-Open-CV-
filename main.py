from src.predict import predict_image

if __name__ == "__main__":

    image_path = input("Enter image path: ")

    result = predict_image(image_path)

    print("Parking Status:", result)