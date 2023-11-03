package com.test.tf;

import org.tensorflow.*;
import org.tensorflow.ndarray.Shape;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Constant;
import org.tensorflow.op.core.Reshape;
import org.tensorflow.op.image.DecodeJpeg;
import org.tensorflow.op.image.ResizeBilinear;
import org.tensorflow.op.io.ReadFile;
import org.tensorflow.types.TFloat32;
import org.tensorflow.types.TString;
import org.tensorflow.types.TUint8;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.util.HashMap;
import java.util.Map;

public class ImageClassification {
    private static final String IC_MODEL_PATH = "C:/Users/USER/Documents/Teravin/DST/bin/model/category_new_brunei_holo_trained_model_v002/";

    private final static String[] cocoLabels = new String[]{
            "fake",
            "real"
    };

    public void classify(Tensor tensor) {
        SavedModelBundle model = SavedModelBundle.load(IC_MODEL_PATH, "serve");
        try (Graph graph = new Graph(); Session session = new Session(graph)) {
            Ops tf = Ops.create(graph);
            Operand<TUint8> uint8Tensor = tf.constant((TUint8) tensor);
            Operand<TFloat32> scaledFloat32Tensor = tf.math.div(tf.dtypes.cast(uint8Tensor, TFloat32.class), tf.constant(255.0f));

            ResizeBilinear resize = tf.image.resizeBilinear(scaledFloat32Tensor, tf.constant(new int[]{64, 64}));

            Reshape<TFloat32> reshape = tf.reshape(resize,
                    tf.array(-1, 64, 64, 3)
            );

            try (TFloat32 reshapeTensor = (TFloat32) session.runner().fetch(reshape).run().get(0)) {

                model.signatures().listIterator().forEachRemaining(System.out::println);

                Map<String, Tensor> feedDict = new HashMap<>();
                feedDict.put("conv2d_2_input", reshapeTensor);
                Result outputTensorMap = model.function("serving_default").call(feedDict);

                try(TFloat32 prediction = (TFloat32) outputTensorMap.get("dense_3").get()) {

                    int classIndex = (prediction.getFloat(0, 0) > prediction.getFloat(0, 1)) ? 0 : 1;

                    String predictedClass = argmax(classIndex);
                    float classifications = Math.abs(prediction.getFloat(0, classIndex) - ((classIndex == 0) ? 1 : 0));
                    System.out.printf("Predicted class: %f%n", prediction.getFloat(0, 0));
                    System.out.printf("Predicted class: %f%n", prediction.getFloat(0, 1));
                    System.out.printf("Predicted class: %f%n", classifications < 0.01 ? 0.01 : classifications);
                    System.out.printf("Predicted class index: %s%n", predictedClass);

                    try {
                        // Get the shape and data from the tensor
                        Shape shape = reshapeTensor.shape();

                        // Convert the tensor to a BufferedImage
                        BufferedImage image = new BufferedImage((int) shape.get(2), (int) shape.get(1), BufferedImage.TYPE_INT_RGB);
                        for (int y = 0; y < shape.get(1); y++) {
                            for (int x = 0; x < shape.get(2); x++) {
                                float r = reshapeTensor.getFloat(0, y, x, 0);
                                float g = reshapeTensor.getFloat(0, y, x, 1);
                                float b = reshapeTensor.getFloat(0, y, x, 2);
                                int rgb = ((int) (r * 255) << 16) | ((int) (g * 255) << 8) | (int) (b * 255);
                                image.setRGB(x, y, rgb);
                            }
                        }

                        // Save the image to file
                        ImageIO.write(image, "jpg", new File(String.format("C:/Users/USER/Documents/TEST/new/%s_%f.jpg", predictedClass, classifications)));
                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                }
            }
        }
    }

    public void classify(String imagePath) {
        SavedModelBundle model = SavedModelBundle.load(IC_MODEL_PATH, "serve");
        try (Graph graph = new Graph(); Session session = new Session(graph)) {
            Ops tf = Ops.create(graph);
            Constant<TString> fileName = tf.constant(imagePath);
            ReadFile readFile = tf.io.readFile(fileName);

            DecodeJpeg.Options options = DecodeJpeg.channels(3L);
            DecodeJpeg decodeImage = tf.image.decodeJpeg(readFile.contents(), options);

            Shape imageShape = session.runner().fetch(decodeImage).run().get(0).shape();

            Reshape<TUint8> reshape = tf.reshape(decodeImage,
                    tf.array(1,
                            imageShape.asArray()[0],
                            imageShape.asArray()[1],
                            imageShape.asArray()[2]
                    )
            );

            try (TUint8 reshapeTensor = (TUint8) session.runner().fetch(reshape).run().get(0)) {

                Operand<TUint8> uint8Tensor = tf.constant(reshapeTensor);
                Operand<TFloat32> scaledFloat32Tensor = tf.math.div(tf.dtypes.cast(uint8Tensor, TFloat32.class), tf.constant(255.0f));

                try (TFloat32 reshapeTensor2 = (TFloat32) session.runner().fetch(scaledFloat32Tensor).run().get(0)) {

                    model.signatures().listIterator().forEachRemaining(System.out::println);

                    Map<String, Tensor> feedDict = new HashMap<>();
                    feedDict.put("conv2d_2_input", reshapeTensor2);
                    Result outputTensorMap = model.function("serving_default").call(feedDict);

                    try(TFloat32 prediction = (TFloat32) outputTensorMap.get("dense_3").get()) {

                        int classIndex = (prediction.getFloat(0, 0) > prediction.getFloat(0, 1)) ? 0 : 1;

                        String predictedClass = argmax(classIndex);
                        float classifications = Math.abs(prediction.getFloat(0, classIndex) - ((classIndex == 0) ? 1 : 0));
                        System.out.printf("Predicted class: %f%n", prediction.getFloat(0, 0));
                        System.out.printf("Predicted class: %f%n", prediction.getFloat(0, 1));
                        System.out.printf("Predicted class: %f%n", classifications < 0.01 ? 0.01 : classifications);
                        System.out.printf("Predicted class index: %s%n", predictedClass);

                        try {
                            // Get the shape and data from the tensor
                            Shape shape = reshapeTensor2.shape();

                            // Convert the tensor to a BufferedImage
                            BufferedImage image = new BufferedImage((int) shape.get(2), (int) shape.get(1), BufferedImage.TYPE_INT_RGB);
                            for (int y = 0; y < shape.get(1); y++) {
                                for (int x = 0; x < shape.get(2); x++) {
                                    float r = reshapeTensor2.getFloat(0, y, x, 0);
                                    float g = reshapeTensor2.getFloat(0, y, x, 1);
                                    float b = reshapeTensor2.getFloat(0, y, x, 2);
                                    int rgb = ((int) (r * 255) << 16) | ((int) (g * 255) << 8) | (int) (b * 255);
                                    image.setRGB(x, y, rgb);
                                }
                            }

                            // Save the image to file
                            ImageIO.write(image, "jpg", new File(String.format("C:/Users/USER/Documents/TEST/new/%s_%f.jpg", predictedClass, classifications)));
                        } catch (Exception e) {
                            e.printStackTrace();
                        }
                    }
                }
            }
        }
    }

    private String argmax(int classIndex) {
        return cocoLabels[classIndex];
    }
}
