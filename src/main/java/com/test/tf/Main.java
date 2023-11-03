package com.test.tf;

public class Main {

    private static String IMAGE_PATH = "C:\\Users\\USER\\Documents\\TEST\\UAT\\IC\\new\\color/";

    public static void main(String[] args) {
        String imagePath = IMAGE_PATH + "image2_01075192_1698042614571.jpg";

        System.out.println("Perform Multi Classification");
        MultiClassification multiClassification = new MultiClassification();
        multiClassification.classify(imagePath);
        System.out.println("Multi Classification Performed");

//		System.out.println("Perform Object Detection");
//		ObjectDetectionService objectDetectionService = new ObjectDetectionService();
//		Tensor tensor = objectDetectionService.detect(imagePath);
//		System.out.println("Object Detection Performed");
//
//		System.out.println("Perform Image Classification");
//		ImageClassification imageClassification = new ImageClassification();
//		imageClassification.classify(tensor);
//		System.out.println("Image Classification Performed");

//		System.out.println("Perform Image Classification");
//		ImageClassification imageClassification = new ImageClassification();
//		imageClassification.classify(RES+);
//		System.out.println("Image Classification Performed");
    }
}