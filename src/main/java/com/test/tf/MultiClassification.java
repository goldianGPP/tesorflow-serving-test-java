package com.test.tf;

import org.tensorflow.*;
import org.tensorflow.ndarray.Shape;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Constant;
import org.tensorflow.types.TFloat32;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

public class MultiClassification {
    private static final String IC_MODEL_PATH = "C:/Users/USER/Documents/Teravin/DST/bin/model/multi_ic_color_trained_model_v008";

    private final static String[] cocoLabels = new String[]{
            "bw",
            "green",
            "red",
            "yellow",
    };

    public void classify(String imagePath) {
        SavedModelBundle model = SavedModelBundle.load(IC_MODEL_PATH, "serve");
        try (Graph graph = new Graph(); Session session = new Session(graph)) {
            Ops tf = Ops.create(graph);

            File imgFile = new File(imagePath);
            BufferedImage img = ImageIO.read(imgFile);

            int newWidth = 224;
            int newHeight = 224;
            BufferedImage resizedImg = new BufferedImage(newWidth, newHeight, img.getType());
            resizedImg.getGraphics().drawImage(img, 0, 0, newWidth, newHeight, null);

            // Convert the image to a float array
            float[][][][] imgArray = new float[1][newWidth][newHeight][3];
            for (int x = 0; x < newWidth; x++) {
                for (int y = 0; y < newHeight; y++) {
                    int rgb = resizedImg.getRGB(x, y);
                    imgArray[0][x][y][0] = ((rgb >> 16) & 0xFF) / 255.0f; // Red channel
                    imgArray[0][x][y][1] = ((rgb >> 8) & 0xFF) / 255.0f;  // Green channel
                    imgArray[0][x][y][2] = (rgb & 0xFF) / 255.0f;         // Blue channel
                }
            }

            Constant<TFloat32> tFloat32Constant = tf.constant(imgArray);

            try (TFloat32 reshapeTensor = (TFloat32) session.runner().fetch(tFloat32Constant).run().get(0)) {

                model.signatures().listIterator().forEachRemaining(System.out::println);

                Map<String, Tensor> feedDict = new HashMap<>();
                feedDict.put("input_2", reshapeTensor);
                Result outputTensorMap = model.function("serving_default").call(feedDict);

                try (TFloat32 prediction = (TFloat32) outputTensorMap.get("dense_2").get()) {
                    int index = argmax(prediction);

                    float score = prediction.getFloat(0, index) * 100;
                    String className = cocoLabels[index];

                    System.out.printf("Predicted class: %f%n", score);
                    System.out.printf("Predicted class name: %s%n", className);

                    if(false)
                        saveImage(reshapeTensor, className, score);
                } catch (IOException e) {
                    throw new RuntimeException(e);
                }
            }
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    private void saveImage(TFloat32 tensor, String predictedClass, float score) throws IOException {
        Shape shape = tensor.shape();
        BufferedImage image = new BufferedImage((int) shape.get(2), (int) shape.get(1), BufferedImage.TYPE_INT_RGB);
        for (int y = 0; y < shape.get(1); y++) {
            for (int x = 0; x < shape.get(2); x++) {
                float r = tensor.getFloat(0, y, x, 0);
                float g = tensor.getFloat(0, y, x, 1);
                float b = tensor.getFloat(0, y, x, 2);
                int rgb = ((int) (r * 255) << 16) | ((int) (g * 255) << 8) | (int) (b * 255);
                image.setRGB(x, y, rgb);
            }
        }

        ImageIO.write(image, "jpg", new File(String.format("C:/Users/USER/Documents/TEST/new/%s_%f.jpg", predictedClass, score)));
    }

    private int argmax(TFloat32 prediction) {
        int size = (int) prediction.size();

        int index = 0;
        float maxScore = prediction.getFloat(0, 0);
        System.out.println("start score : " + maxScore);
        for (int i = 1; i < size; i++) {
            float score = prediction.getFloat(0, i);
            System.out.println(i + ". score : " + score);
            if (score > maxScore) {
                maxScore = score;
                index = i;
            }
        }

        System.out.println("result maxScore : " + prediction.getFloat(0, index));
        return index;
    }
}
